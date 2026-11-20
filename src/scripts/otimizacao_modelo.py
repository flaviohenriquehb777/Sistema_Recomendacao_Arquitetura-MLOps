"""
Estágio de treinamento do recomendador (pipeline DVC).

Treina o recomendador neural **NeuMF** (GMF + MLP) com *objetivo de ranking*:
classificação binária de interação cliente-item com **amostragem de negativos**,
em vez de regressão sobre ``Sales``.

Por que ranking e não regressão?
    Prever ``Sales`` normalizado produz R²≈0 e scores quase constantes — o modelo
    praticamente não ordena o catálogo. O objetivo de um sistema de recomendação é
    separar itens relevantes de irrelevantes; isso exige uma perda de ranking
    (aqui, entropia cruzada binária sobre pares positivos/negativos).

Protocolo (leave-last-out, estilo NCF):
    * **teste** = último item distinto de cada cliente;
    * **validação** = penúltimo item;
    * **treino** = os demais.
    O treino é encerrado pelo ``RankingCheckpoint``, que monitora o **NDCG@k de
    validação** (a métrica-alvo) e restaura os melhores pesos.

Modelo final — **híbrido popularity-first**:
    ``score = z(log1p(popularidade)) + alpha * z(NeuMF)``. O prior de popularidade
    (contado no treino) é o sinal-base e o NeuMF entra como ajuste ponderado por
    ``alpha``, escolhido na validação com política conservadora (só adota ``alpha>0``
    se ele superar o puramente popular por uma margem mínima). As métricas finais
    são reportadas no **teste**, ranqueando o catálogo completo e excluindo os itens
    já vistos, comparadas aos baselines aleatório e mais-populares.

Saídas (outs do DVC):
    models/best_model_recomendacao_otimizado.keras
    models/hybrid_config_otimizado.pkl  (alpha + prior de popularidade)
    models/customer_encoder_otimizado.pkl
    models/product_encoder_otimizado.pkl
    models/category_encoder_otimizado.pkl
    models/subcategory_encoder_otimizado.pkl

Uso:
    python -m src.scripts.otimizacao_modelo
"""

import joblib
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from src.config.mlflow_tracking import log_run, setup_mlflow
from src.config.model_utils import (
    RANDOM_SEED,
    TOP_K,
    RankingCheckpoint,
    build_item_catalog,
    build_training_pairs,
    compute_popularity_prior,
    create_ranking_model,
    evaluate_hybrid,
    evaluate_most_popular,
    evaluate_random_baseline,
    evaluate_ranking,
    recommend_products,
    select_alpha,
    to_model_inputs,
    train_ranking_model,
    train_validation_test_split,
)
from src.config.paths import DADOS_TRATADOS, MODELS_DIR

# Hiperparâmetros do recomendador (escolhidos por validação — ver docstring de
# create_ranking_model sobre a necessidade de regularização forte neste dado).
EMBEDDING_DIM = 32
L2_STRENGTH = 1e-3
DROPOUT = 0.3
N_NEGATIVES = 8
EPOCHS = 40
BATCH_SIZE = 256
PATIENCE = 5

COLUNAS = ["Customer Name", "Product ID", "Product Name", "Category", "Sub-Category", "Order Date"]


def carregar_e_codificar_dados():
    """Carrega a base tratada e ajusta os codificadores das variáveis categóricas."""
    df = pd.read_parquet(DADOS_TRATADOS)[COLUNAS].copy()
    encoders = {
        "customer": LabelEncoder().fit(df["Customer Name"]),
        "product": LabelEncoder().fit(df["Product ID"]),
        "category": LabelEncoder().fit(df["Category"]),
        "subcategory": LabelEncoder().fit(df["Sub-Category"]),
    }
    return df, encoders


def imprimir_metricas(titulo: str, metricas: dict) -> None:
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
    for nome, valor in metricas.items():
        if isinstance(valor, float):
            print(f"  {nome:<16}: {valor:.4f}")
        else:
            print(f"  {nome:<16}: {int(valor)}")
    print("=" * 70)


def main() -> None:
    print("=" * 70)
    print("TREINAMENTO DO RECOMENDADOR (OBJETIVO DE RANKING — NeuMF)")
    print("=" * 70)

    print("Configurando MLflow...")
    setup_mlflow()

    df, encoders = carregar_e_codificar_dados()
    print(f"Base carregada: {df.shape[0]} interações, "
          f"{df['Customer Name'].nunique()} clientes, {df['Product ID'].nunique()} produtos.")

    # --- Catálogo de itens (alinhado ao índice do encoder de produtos) --------
    catalog = build_item_catalog(
        df, encoders["product"], encoders["category"], encoders["subcategory"]
    )
    num_produtos = len(catalog[0])
    num_clientes = len(encoders["customer"].classes_)
    num_categorias = len(encoders["category"].classes_)
    num_subcategorias = len(encoders["subcategory"].classes_)
    print(f"Dimensões: {num_clientes} clientes | {num_produtos} produtos | "
          f"{num_categorias} categorias | {num_subcategorias} sub-categorias")

    # --- Divisão leave-last-out (treino / validação / teste) -----------------
    train_pairs, val_pairs, test_pairs = train_validation_test_split(
        df, encoders["customer"], encoders["product"]
    )
    print(f"\nSplit: {len(train_pairs)} pares de treino | "
          f"{len(val_pairs)} clientes na validação | {len(test_pairs)} no teste.")

    # --- Conjuntos com amostragem de negativos -------------------------------
    tc, tp, ty = build_training_pairs(
        train_pairs, num_produtos, n_negatives=N_NEGATIVES, seed=RANDOM_SEED
    )
    vc, vp, vy = build_training_pairs(
        val_pairs, num_produtos, n_negatives=4, seed=RANDOM_SEED, exclude_pairs=train_pairs
    )
    print(f"Treino: {len(ty)} exemplos ({int(ty.sum())} positivos / {int((ty == 0).sum())} negativos).")

    # --- Modelo --------------------------------------------------------------
    modelo = create_ranking_model(
        num_clientes, num_produtos, num_categorias, num_subcategorias,
        embedding_dim=EMBEDDING_DIM, l2_strength=L2_STRENGTH,
        dropout_rate=DROPOUT, seed=RANDOM_SEED,
    )
    modelo.summary()

    ranking_checkpoint = RankingCheckpoint(
        val_pairs, train_pairs, catalog, k=TOP_K, patience=PATIENCE, verbose=1
    )
    callbacks = [
        ranking_checkpoint,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
    ]
    history = train_ranking_model(
        modelo, tc, tp, catalog[1][tp], catalog[2][tp], ty,
        validation_data=(to_model_inputs(vc, vp, catalog), vy),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2,
    )
    print(f"\nTreinamento encerrado em {len(history.history['loss'])} épocas. "
          f"Melhor NDCG@{TOP_K} de validação: {ranking_checkpoint.best:.5f} "
          f"(época {ranking_checkpoint.best_epoch}).")

    # --- Prior de popularidade e peso do sinal personalizado (validação) -----
    popularity = compute_popularity_prior(train_pairs, num_produtos)
    alpha, historico_alpha = select_alpha(
        modelo, val_pairs, train_pairs, catalog, popularity, k=TOP_K
    )
    print(f"\nCalibração do híbrido na validação (NDCG@{TOP_K}):")
    for valor, ndcg in historico_alpha.items():
        print(f"  alpha={valor:<4} -> val_ndcg@{TOP_K}={ndcg:.5f}")
    print(f"  alpha escolhido: {alpha}")

    # --- Avaliação final no TESTE (catálogo completo, exclui vistos) ---------
    metricas_hybrid = evaluate_hybrid(
        modelo, test_pairs, train_pairs, catalog, popularity, alpha, k=TOP_K
    )
    metricas_modelo = evaluate_ranking(modelo, test_pairs, train_pairs, catalog, k=TOP_K)
    metricas_populares = evaluate_most_popular(test_pairs, train_pairs, num_produtos, k=TOP_K)
    metricas_aleatorio = evaluate_random_baseline(test_pairs, train_pairs, num_produtos, k=TOP_K)

    imprimir_metricas(f"TESTE — HÍBRIDO popularity-first (alpha={alpha})", metricas_hybrid)
    imprimir_metricas("TESTE — NeuMF puro (alpha=0)", metricas_modelo)
    imprimir_metricas("TESTE — BASELINE MAIS-POPULARES", metricas_populares)
    imprimir_metricas("TESTE — BASELINE ALEATÓRIO (esperança analítica)", metricas_aleatorio)

    ganho_pop = metricas_hybrid[f"ndcg@{TOP_K}"] / metricas_populares[f"ndcg@{TOP_K}"] - 1
    print(f"\nGanho de NDCG@{TOP_K} do híbrido sobre o baseline mais-populares: {ganho_pop:+.1%}")

    # --- Registro no MLflow (não-bloqueante) ---------------------------------
    metricas_log = {
        **metricas_hybrid,
        "alpha": float(alpha),
        "val_ndcg": ranking_checkpoint.best,
        "epocas_treinadas": float(len(history.history["loss"])),
        "ndcg_neumf_puro": metricas_modelo[f"ndcg@{TOP_K}"],
        "ndcg_baseline_populares": metricas_populares[f"ndcg@{TOP_K}"],
        "ganho_ndcg_sobre_populares": float(ganho_pop),
    }
    params_log = {
        "base": "sc_base_01",
        "embedding_dim": EMBEDDING_DIM,
        "l2_strength": L2_STRENGTH,
        "dropout": DROPOUT,
        "n_negatives": N_NEGATIVES,
        "epochs_max": EPOCHS,
        "batch_size": BATCH_SIZE,
        "patience": PATIENCE,
        "top_k": TOP_K,
        "num_clientes": num_clientes,
        "num_produtos": num_produtos,
    }
    run_id = log_run(params_log, metricas_log, run_name="producao_neumf_hibrido")
    if run_id:
        print(f"Registrado no MLflow. Run ID: {run_id}")
    else:
        print("Tracking MLflow indisponível — treino seguiu normalmente.")

    # --- Persistência dos artefatos -----------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    caminho_modelo = MODELS_DIR / "best_model_recomendacao_otimizado.keras"
    modelo.save(caminho_modelo)
    print(f"\nModelo salvo em: {caminho_modelo}")

    caminho_hybrid = MODELS_DIR / "hybrid_config_otimizado.pkl"
    joblib.dump({"alpha": float(alpha), "popularity": popularity}, caminho_hybrid)
    print(f"Config do híbrido salva em: {caminho_hybrid}")

    for nome, encoder in encoders.items():
        caminho = MODELS_DIR / f"{nome}_encoder_otimizado.pkl"
        joblib.dump(encoder, caminho)
        print(f"Encoder salvo em: {caminho}")

    # --- Demonstração de recomendação ---------------------------------------
    cliente_demo = df["Customer Name"].iloc[0]
    recomendacoes = recommend_products(
        modelo, cliente_demo, df, encoders["customer"], encoders["product"], catalog,
        top_k=TOP_K, popularity=popularity, alpha=alpha,
    )
    print("\n" + "=" * 70)
    print("RECOMENDAÇÕES PARA: cliente de exemplo (identificador anonimizado)")
    print("=" * 70)
    print(recomendacoes.to_string(index=False))
    print("=" * 70)

    print("\nTREINAMENTO CONCLUÍDO COM SUCESSO.")


if __name__ == "__main__":
    main()
