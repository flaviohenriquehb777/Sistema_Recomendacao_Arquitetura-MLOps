"""Testes da régua de avaliação reforçada (``src/config/avaliacao_ranking.py``).

Usa uma fixture mínima, construída em memória, com estrutura de co-ocorrência
conhecida, para checar cada componente sem depender de dados versionados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.avaliacao_ranking import (
    _metricas_linha,
    _ndcg,
    _topk_com_exclusao,
    avaliar_scorer,
    intervalo_confianca,
    matrizes_do_split,
    scorer_constante,
    scorer_fatores,
    scorer_item_knn,
    treinar_item_knn,
)

N_ITENS = 10


def _pares(linhas: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(linhas, columns=["c", "p"])


def _fixture_treino() -> pd.DataFrame:
    """Item 9 é raro, mas co-ocorre com o item 0 — só o item-kNN o enxerga."""
    return _pares(
        [
            (0, 0),
            (1, 0),
            (1, 9),
            (2, 1),
            (2, 2),
            (2, 3),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 1),
            (4, 2),
            (4, 3),
        ]
    )


def _fixture_avaliacao() -> pd.DataFrame:
    return _pares([(0, 9), (1, 1), (2, 1), (3, 2), (4, 3)])


def test_topk_com_exclusao_ignora_vistos() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    top = _topk_com_exclusao(scores, excluidos=[0], k=2)
    assert top.tolist() == [1, 2]


def test_ndcg_ordenacao_perfeita_e_unitaria() -> None:
    assert _ndcg(np.array([1.0, 1.0, 0.0])) == pytest.approx(1.0)
    assert _ndcg(np.array([])) == 0.0


def test_intervalo_confianca_contem_a_media() -> None:
    valores = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ic = intervalo_confianca(valores, n_boot=200, seed=42)
    assert ic["ic_inferior"] <= ic["media"] <= ic["ic_superior"]
    assert ic["media"] == pytest.approx(2.0)


def test_item_knn_encontra_coocorrencia_rara() -> None:
    treino = _fixture_treino()
    matriz, vistos, _ = matrizes_do_split(treino, _fixture_avaliacao(), 5, N_ITENS)
    similaridade = treinar_item_knn(matriz, top_vizinhos=5)
    scorer = scorer_item_knn(matriz, similaridade)

    scores_u0 = scorer(np.array([0]))[0]
    top = _topk_com_exclusao(scores_u0, vistos[0], k=1)
    assert top.tolist() == [9]


def test_popularidade_recomenda_os_frequentes() -> None:
    popularidade = np.array([2, 4, 3, 3, 0, 0, 0, 0, 0, 1], dtype="float64")
    scorer = scorer_constante(popularidade)
    scores = scorer(np.array([0, 1]))
    assert np.argmax(scores[0]) == 1  # item mais frequente
    assert scores.shape == (2, N_ITENS)


def test_scorer_fatores_produto_interno() -> None:
    usuarios = np.array([[1.0, 0.0]])
    itens = np.array([[1.0, 0.0], [0.0, 1.0]])
    scorer = scorer_fatores(usuarios, itens)
    assert scorer(np.array([0])).tolist() == [[1.0, 0.0]]


def test_avaliar_scorer_reporta_metricas_catalogo_e_ic() -> None:
    treino = _fixture_treino()
    avaliacao = _fixture_avaliacao()
    matriz, vistos, relevantes = matrizes_do_split(treino, avaliacao, 5, N_ITENS)
    popularidade = np.asarray(matriz.sum(axis=0)).ravel()
    categorias = np.array([i % 3 for i in range(N_ITENS)])

    similaridade = treinar_item_knn(matriz, top_vizinhos=5)
    resultado = avaliar_scorer(
        scorer_item_knn(matriz, similaridade),
        np.arange(5),
        vistos,
        relevantes,
        n_itens=N_ITENS,
        popularidade_treino=popularidade,
        categorias_por_item=categorias,
        k=3,
        n_boot=50,
    )

    assert "ndcg@3" in resultado
    assert "cobertura_catalogo" in resultado
    assert "novidade" in resultado
    assert "diversidade_categoria" in resultado
    assert resultado["ndcg@3_ic"]["ic_inferior"] <= resultado["ndcg@3"]


def test_matrizes_do_split_tem_formato_correto() -> None:
    matriz, vistos, relevantes = matrizes_do_split(
        _fixture_treino(), _fixture_avaliacao(), 5, N_ITENS
    )
    assert matriz.shape == (5, N_ITENS)
    assert vistos[1] == {0, 9}
    assert relevantes[0] == {9}


def test_metricas_linha_conta_acerto() -> None:
    scores = np.zeros(N_ITENS)
    scores[9] = 1.0
    rec, ndcg, map_k, hit = _metricas_linha(scores, vistos=[0], relevantes={9}, k=3)
    assert rec[0] == 9
    assert ndcg == pytest.approx(1.0)
    assert map_k == pytest.approx(1.0)
    assert hit == 1.0
