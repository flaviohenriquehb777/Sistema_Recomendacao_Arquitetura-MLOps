#!/usr/bin/env python3
"""
Benchmark sobre a base pública REAL **sc_base_03** (e-commerce, feedback
implícito anonimizado) — roda o pipeline de recomendação REAL e reporta métricas
reais.

Motivação
---------
A base própria (sc_base_01) é pequena e esparsa demais para um modelo neural
superar a popularidade. A base sintética demonstra a arquitetura, mas é, por
construção, circular. A sc_base_03 fecha essa lacuna: é um **log de eventos
real** de um site de e-commerce, com sinal implícito de verdade (``view``,
``addtocart``, ``transaction``) — o cenário em que um recomendador prova valor.

O que este script faz
---------------------
    1. converte os CSVs brutos no esquema padrão (ver ``sc_base_03_data``);
    2. treina a MESMA NeuMF, com o MESMO protocolo (leave-last-out, negativos,
       early stopping por NDCG@k de validação, híbrido popularity-first);
    3. reporta as métricas no **teste**, comparadas aos baselines (mais-populares
       e aleatório).

Honestidade metodológica
------------------------
A base é **real**, porém **anonimizada** (sem nomes de produto/cliente) e aplica
**filtros explícitos de tratabilidade** (itens/usuários relevantes + amostra
determinística) — ambos documentados em ``sc_base_03_data``. Como os dados são
reais, **não existe "teto oracle"**: não há fator latente verdadeiro conhecido.

Saídas:
    dados/benchmark_sc_base_03.parquet          (base preparada, reproduzível)
    report/benchmark_sc_base_03_metricas.json   (métricas + metadados)

Uso:
    python -m src.scripts.benchmark_sc_base_03
"""

import json

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
    select_alpha,
    to_model_inputs,
    train_ranking_model,
    train_validation_test_split,
)
from src.config.paths import DATA_DIR, REPORTS_DIR
from src.config.sc_base_03_data import construir_interacoes, resumo_base

# --- Filtros de tratabilidade da base bruta (ver sc_base_03_data) ----------
MIN_EVENTOS_ITEM = 100
MIN_ITENS_USUARIO = 5
MAX_CLIENTES = 5000

# --- Hiperparâmetros do modelo (arquitetura idêntica à de produção) ----------
# Os dados reais têm sinal colaborativo; como no benchmark de escala, NÃO se
# aplica a regularização forte calibrada para a base minúscula do sc_base_01.
EMBEDDING_DIM = 32
L2_STRENGTH = 0.0
DROPOUT = 0.0
N_NEGATIVES = 8
EPOCHS = 40
BATCH_SIZE = 512
PATIENCE = 5

COLUNAS = ["Customer Name", "Product ID", "Product Name", "Category", "Sub-Category", "Order Date"]


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
    print("BENCHMARK SOBRE BASE PÚBLICA REAL — sc_base_03 (NeuMF + híbrido)")
    print("=" * 70)

    # Tracking MLflow (não-bloqueante). Experimento PRÓPRIO do benchmark público,
    # para separar esta demonstração (dado real) do benchmark sintético e do
    # modelo de produção.
    setup_mlflow(experiment_name="recomendacao_benchmark_publico")

    # --- Base sc_base_03 (preparada a partir do bruto do portal de dados) -----------
    print("\nPreparando a base a partir dos CSVs brutos da sc_base_03...")
    df = construir_interacoes(
        min_eventos_item=MIN_EVENTOS_ITEM,
        min_itens_usuario=MIN_ITENS_USUARIO,
        max_clientes=MAX_CLIENTES,
        seed=RANDOM_SEED,
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    caminho_base = DATA_DIR / "benchmark_sc_base_03.parquet"
    df = df[COLUNAS]
    df.to_parquet(caminho_base, index=False)
    print(f"Base preparada e salva em: {caminho_base}")

    resumo = resumo_base(df)
    print("\nResumo da base sc_base_03 (após filtros):")
    for chave, valor in resumo.items():
        print(f"  {chave:<26}: {valor}")

    encoders = {
        "customer": LabelEncoder().fit(df["Customer Name"]),
        "product": LabelEncoder().fit(df["Product ID"]),
        "category": LabelEncoder().fit(df["Category"]),
        "subcategory": LabelEncoder().fit(df["Sub-Category"]),
    }

    catalog = build_item_catalog(
        df, encoders["product"], encoders["category"], encoders["subcategory"]
    )
    num_produtos = len(catalog[0])
    num_clientes = len(encoders["customer"].classes_)
    num_categorias = len(encoders["category"].classes_)
    num_subcategorias = len(encoders["subcategory"].classes_)
    print(f"\nDimensões: {num_clientes} clientes | {num_produtos} produtos | "
          f"{num_categorias} categorias | {num_subcategorias} sub-categorias")

    train_pairs, val_pairs, test_pairs = train_validation_test_split(
        df, encoders["customer"], encoders["product"]
    )
    print(f"Split: {len(train_pairs)} pares de treino | "
          f"{len(val_pairs)} clientes na validação | {len(test_pairs)} no teste.")

    tc, tp, ty = build_training_pairs(
        train_pairs, num_produtos, n_negatives=N_NEGATIVES, seed=RANDOM_SEED
    )
    vc, vp, vy = build_training_pairs(
        val_pairs, num_produtos, n_negatives=4, seed=RANDOM_SEED, exclude_pairs=train_pairs
    )
    print(f"Treino: {len(ty)} exemplos ({int(ty.sum())} positivos / {int((ty == 0).sum())} negativos).")

    # --- Treino --------------------------------------------------------------
    modelo = create_ranking_model(
        num_clientes, num_produtos, num_categorias, num_subcategorias,
        embedding_dim=EMBEDDING_DIM, l2_strength=L2_STRENGTH,
        dropout_rate=DROPOUT, seed=RANDOM_SEED,
    )
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

    # --- Híbrido + avaliação final no teste ----------------------------------
    popularity = compute_popularity_prior(train_pairs, num_produtos)
    alpha, historico_alpha = select_alpha(
        modelo, val_pairs, train_pairs, catalog, popularity, k=TOP_K
    )
    print(f"\nCalibração do híbrido na validação (NDCG@{TOP_K}):")
    for valor, ndcg in historico_alpha.items():
        print(f"  alpha={valor:<4} -> val_ndcg@{TOP_K}={ndcg:.5f}")
    print(f"  alpha escolhido: {alpha}")

    metricas_hybrid = evaluate_hybrid(
        modelo, test_pairs, train_pairs, catalog, popularity, alpha, k=TOP_K
    )
    metricas_modelo = evaluate_ranking(modelo, test_pairs, train_pairs, catalog, k=TOP_K)
    metricas_populares = evaluate_most_popular(test_pairs, train_pairs, num_produtos, k=TOP_K)
    metricas_aleatorio = evaluate_random_baseline(test_pairs, train_pairs, num_produtos, k=TOP_K)

    # Seleção do modelo final **por validação** (NDCG@k): NeuMF puro x híbrido.
    val_neumf = float(ranking_checkpoint.best)
    val_hibrido = float(historico_alpha[alpha])
    if val_neumf >= val_hibrido:
        nome_final, metricas_final, val_final = "NeuMF puro", metricas_modelo, val_neumf
    else:
        nome_final = f"híbrido popularity-first (alpha={alpha})"
        metricas_final, val_final = metricas_hybrid, val_hibrido

    imprimir_metricas(f"TESTE — MODELO FINAL: {nome_final}", metricas_final)
    imprimir_metricas("TESTE — ablação: híbrido popularity-first", metricas_hybrid)
    imprimir_metricas("TESTE — ablação: NeuMF puro", metricas_modelo)
    imprimir_metricas("TESTE — BASELINE MAIS-POPULARES", metricas_populares)
    imprimir_metricas("TESTE — BASELINE ALEATÓRIO (esperança analítica)", metricas_aleatorio)

    ndcg_final = metricas_final[f"ndcg@{TOP_K}"]
    ndcg_pop = metricas_populares[f"ndcg@{TOP_K}"]
    ganho_pop = ndcg_final / ndcg_pop - 1 if ndcg_pop else float("nan")

    print(f"\nModelo final: {nome_final} (val_ndcg@{TOP_K}={val_final:.5f})")
    print(f"Ganho de NDCG@{TOP_K} sobre o baseline mais-populares: {ganho_pop:+.1%}")
    print(f"NDCG@{TOP_K}: final={ndcg_final:.4f} | mais-populares={ndcg_pop:.4f}")

    # --- Persistência das métricas -------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    saida = {
        "base": "sc_base_03 (publica, real, anonimizada)",
        "filtros": {
            "min_eventos_item": MIN_EVENTOS_ITEM,
            "min_itens_usuario": MIN_ITENS_USUARIO,
            "max_clientes": MAX_CLIENTES,
        },
        "resumo_base": resumo,
        "alpha_hibrido": float(alpha),
        "modelo_final": nome_final,
        "val_ndcg_final": float(val_final),
        "val_ndcg_neumf": float(val_neumf),
        "metricas_teste": {
            "final": metricas_final,
            "hibrido": metricas_hybrid,
            "neumf_puro": metricas_modelo,
            "mais_populares": metricas_populares,
            "aleatorio": metricas_aleatorio,
        },
        "ganho_ndcg_sobre_populares": float(ganho_pop),
    }
    caminho_json = REPORTS_DIR / "benchmark_sc_base_03_metricas.json"
    caminho_json.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMétricas salvas em: {caminho_json}")

    # --- Registro no MLflow (não-bloqueante) ---------------------------------
    params_log = {
        "base": "sc_base_03_publica_real",
        "min_eventos_item": MIN_EVENTOS_ITEM,
        "min_itens_usuario": MIN_ITENS_USUARIO,
        "max_clientes": MAX_CLIENTES,
        "n_clientes": resumo["clientes"],
        "n_produtos": resumo["produtos"],
        "embedding_dim": EMBEDDING_DIM,
        "l2_strength": L2_STRENGTH,
        "dropout": DROPOUT,
        "n_negatives": N_NEGATIVES,
        "epochs_max": EPOCHS,
        "batch_size": BATCH_SIZE,
        "patience": PATIENCE,
        "top_k": TOP_K,
        "alpha_hibrido": float(alpha),
        "modelo_final": nome_final,
    }
    metrics_log = {
        **{f"final_{chave}": valor for chave, valor in metricas_final.items()},
        **{f"popular_{chave}": valor for chave, valor in metricas_populares.items()},
        "val_ndcg_final": float(val_final),
        "ganho_ndcg_sobre_populares": float(ganho_pop),
    }
    run_id = log_run(
        params_log, metrics_log, artifacts=[caminho_json],
        run_name="benchmark_publico_sc_base_03",
    )
    if run_id:
        print(f"Registrado no MLflow. Run ID: {run_id}")
    else:
        print("Tracking MLflow indisponível — benchmark seguiu normalmente.")

    print("\nBENCHMARK CONCLUÍDO COM SUCESSO.")


if __name__ == "__main__":
    main()
