#!/usr/bin/env python3
"""
Benchmark de **escala** — roda o pipeline de recomendação REAL sobre uma base
**sintética** (ver ``src/config/synthetic_data.py``) e reporta métricas reais.

Motivação
---------
A base pública pequena (sc_base_01: ~800 clientes, ~12 itens/cliente, densidade
0,66 %) não contém sinal colaborativo suficiente para um modelo neural superar o
*baseline* de popularidade. Para demonstrar o comportamento esperado de um
recomendador em **escala de produção**, este script:

    1. gera uma base sintética **reprodutível** com estrutura latente real;
    2. treina a MESMA NeuMF com o MESMO protocolo (leave-last-out, negativos,
       early stopping por NDCG@k de validação, híbrido popularity-first);
    3. reporta as métricas no **teste**, comparadas aos baselines.

A base é **sintética e rotulada como tal** — não representa dados reais de
nenhuma empresa. O que se demonstra é a **arquitetura e o método**, não "os dados".

Saídas:
    dados/benchmark_sintetico.parquet          (base sintética reproduzível)
    report/benchmark_sintetico_metricas.json   (métricas + metadados)

Uso:
    python -m src.scripts.benchmark_sintetico
"""

import json

import numpy as np
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
    evaluate_from_scores,
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
from src.config.synthetic_data import dataset_summary, generate_synthetic_dataset

# Escala do benchmark (base sintética — ver docstring do gerador).
N_USERS = 3000
N_ITEMS = 1000
AVG_INTERACTIONS = 22

# Hiperparâmetros do modelo (arquitetura idêntica à de produção).
#
# IMPORTANTE (regularização por validação): a pipeline de produção usa L2=1e-3 e
# dropout=0,3 porque a base REAL é minúscula e muito esparsa (~12 itens/cliente
# em ~1.900 produtos) — ali, sem regularização forte, o modelo decora o treino.
# Em uma base com sinal colaborativo real em escala (como esta), a mesma
# regularização **sufoca** a personalização: a validação seleciona um modelo
# praticamente sem regularização, que de fato aprende a preferência latente e
# supera o baseline de popularidade por uma margem grande. Os valores abaixo
# foram escolhidos por NDCG@k de validação (ver `RankingCheckpoint`).
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


def _oracle_metrics(fatores, encoders, train_pairs, test_pairs, k):
    """
    Teto teórico do protocolo: ranqueia cada cliente pela **afinidade latente
    verdadeira** (o sinal que gerou os dados), sem nenhum ruído de aprendizagem.

    Serve para mostrar que a estrutura colaborativa existe e é recuperável, e
    quanto do teto o modelo aprendido de fato captura.
    """
    u_src = np.array([int(nome.split()[1]) for nome in encoders["customer"].classes_], dtype="int64")
    p_src = np.array([int(pid.split("-")[1]) for pid in encoders["product"].classes_], dtype="int64")
    usuarios = fatores["user_latent"][u_src]
    itens = fatores["item_latent"][p_src]
    customers = np.asarray(sorted(test_pairs["c"].unique()), dtype="int64")
    scores = usuarios[customers] @ itens.T
    return evaluate_from_scores(scores, train_pairs, test_pairs, k=k)


def main() -> None:
    print("=" * 70)
    print("BENCHMARK DE ESCALA SOBRE BASE SINTÉTICA (NeuMF + híbrido)")
    print("=" * 70)

    # Tracking MLflow (não-bloqueante). Experimento PRÓPRIO do benchmark, para
    # separar a demonstração de escala (base sintética) do modelo de produção.
    setup_mlflow(experiment_name="recomendacao_benchmark_escala")

    # --- Base sintética (reprodutível) ---------------------------------------
    # Regenera sempre (geração determinística, seed fixa) para manter os fatores
    # latentes verdadeiros em memória — necessários para o teto "oracle". O
    # parquet salvo é o artefato reproduzível do benchmark.
    df, fatores = generate_synthetic_dataset(
        n_users=N_USERS, n_items=N_ITEMS, avg_interactions=AVG_INTERACTIONS,
        return_factors=True,
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    caminho_base = DATA_DIR / "benchmark_sintetico.parquet"
    df.to_parquet(caminho_base, index=False)
    print(f"Base sintética gerada (reprodutível) e salva em: {caminho_base}")

    resumo = dataset_summary(df)
    print("\nResumo da base sintética:")
    for chave, valor in resumo.items():
        print(f"  {chave:<26}: {valor}")

    df = df[COLUNAS].copy()
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
    metricas_oracle = _oracle_metrics(fatores, encoders, train_pairs, test_pairs, TOP_K)

    # Seleção do modelo final **por validação** (NDCG@k): NeuMF puro x híbrido.
    # Em escala, a rede aprende sozinha o sinal de popularidade (viés de item), de
    # modo que o prior explícito pode apenas diluir a personalização — a validação
    # decide qual das duas formulações fica com o modelo final.
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
    imprimir_metricas("TESTE — TETO 'ORACLE' (afinidade latente verdadeira)", metricas_oracle)
    imprimir_metricas("TESTE — BASELINE ALEATÓRIO (esperança analítica)", metricas_aleatorio)

    ndcg_final = metricas_final[f"ndcg@{TOP_K}"]
    ndcg_pop = metricas_populares[f"ndcg@{TOP_K}"]
    ndcg_oracle = metricas_oracle[f"ndcg@{TOP_K}"]
    ganho_pop = ndcg_final / ndcg_pop - 1 if ndcg_pop else float("nan")

    print(f"\nModelo final: {nome_final} (val_ndcg@{TOP_K}={val_final:.5f})")
    print(f"Ganho de NDCG@{TOP_K} sobre o baseline mais-populares: {ganho_pop:+.1%}")
    print(f"NDCG@{TOP_K}: final={ndcg_final:.4f} | mais-populares={ndcg_pop:.4f} | "
          f"oracle(afinidade)={ndcg_oracle:.4f}")

    # --- Persistência das métricas -------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    saida = {
        "base": "sintetica (reprodutivel, seed=42)",
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
            "oracle_afinidade": metricas_oracle,
            "aleatorio": metricas_aleatorio,
        },
        "ganho_ndcg_sobre_populares": float(ganho_pop),
    }
    caminho_json = REPORTS_DIR / "benchmark_sintetico_metricas.json"
    caminho_json.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMétricas salvas em: {caminho_json}")

    # --- Registro no MLflow (não-bloqueante) ---------------------------------
    params_log = {
        "base": "sintetica_reprodutivel",
        "n_users": N_USERS,
        "n_items": N_ITEMS,
        "avg_interactions": AVG_INTERACTIONS,
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
        **{f"oracle_{chave}": valor for chave, valor in metricas_oracle.items()},
        "val_ndcg_final": float(val_final),
        "ganho_ndcg_sobre_populares": float(ganho_pop),
    }
    run_id = log_run(
        params_log, metrics_log, artifacts=[caminho_json],
        run_name="benchmark_escala_sintetico",
    )
    if run_id:
        print(f"Registrado no MLflow. Run ID: {run_id}")
    else:
        print("Tracking MLflow indisponível — benchmark seguiu normalmente.")

    print("\nBENCHMARK CONCLUÍDO COM SUCESSO.")


if __name__ == "__main__":
    main()
