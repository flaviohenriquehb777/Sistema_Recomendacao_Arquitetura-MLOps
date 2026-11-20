"""
Régua de avaliação reforçada de recomendação por ranking.

Complementa ``model_utils`` com o que faltava para uma avaliação honesta em
escala:

    - métricas **por usuário** (NDCG@k, MAP@k, Hit-Rate@k), base para incerteza;
    - métricas de **saúde do catálogo**: cobertura, novidade e diversidade;
    - **intervalo de confiança** por bootstrap sobre os usuários;
    - **baselines fortes escaláveis**, todos sem dependência nova:
      popularidade, recência, item-kNN (co-ocorrência cosseno truncada em
      top-K vizinhos) e ALS implícito (fatoração por mínimos quadrados com
      confiança, resolvida com SciPy).

Tudo opera sobre o MESMO protocolo do treino e do serving: leave-last-out com
exclusão dos itens já vistos e catálogo candidato completo. Os baselines são
"scorers" — funções ``f(índices de usuário) -> matriz (len, n_itens) de scores``
— avaliados pelo mesmo laço, de modo que nenhum deles ganha vantagem de
protocolo.

Princípios:
    - Nada é amostrado de forma que mude a conclusão: a avaliação percorre todos
      os usuários elegíveis; o bootstrap só quantifica a incerteza.
    - Um item visto no treino nunca é recomendado (exclusão explícita).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy import sparse

from src.config.model_utils import RANDOM_SEED, TOP_K

# Tamanho de bloco ao pedir scores ao modelo/baseline (limita o pico de memória:
# bloco x n_itens floats). 128 x 207k floats ~ 212 MB em float64.
TAMANHO_BLOCO = 128
# Número padrão de vizinhos item-item mantidos por item.
TOP_VIZINHOS_KNN = 100


# ---------------------------------------------------------------------------
# Métricas por usuário
# ---------------------------------------------------------------------------
def _topk_com_exclusao(
    scores: np.ndarray, excluidos: Sequence[int], k: int
) -> np.ndarray:
    """Índices dos top-k itens, ignorando os itens já vistos, em ordem de score."""
    scores = np.asarray(scores, dtype="float64")
    if len(excluidos):
        scores[np.asarray(list(excluidos), dtype=np.int64)] = -np.inf
    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    candidatos = np.argpartition(-scores, k - 1)[:k]
    return candidatos[np.argsort(-scores[candidatos], kind="stable")]


def _ndcg(relevancias: np.ndarray) -> float:
    """NDCG de uma lista de ganhos binários na ordem apresentada."""
    if relevancias.size == 0:
        return 0.0
    posicoes = np.arange(2, relevancias.size + 2)
    dcg = np.sum(relevancias / np.log2(posicoes))
    ideal = np.sort(relevancias)[::-1]
    idcg = np.sum(ideal / np.log2(posicoes))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _metricas_linha(
    scores: np.ndarray, vistos: Sequence[int], relevantes: set[int], k: int
) -> tuple[np.ndarray, float, float, float]:
    """Métricas de um usuário + a lista recomendada (para cobertura/novidade)."""
    recomendados = _topk_com_exclusao(scores, vistos, k)
    if recomendados.size == 0:
        return recomendados, 0.0, 0.0, 0.0
    hit = np.isin(recomendados, list(relevantes)).astype("float64")
    ndcg = _ndcg(hit)
    hit_rate = 1.0 if hit.sum() > 0 else 0.0
    acertos, ap = 0.0, 0.0
    for pos in range(recomendados.size):
        if hit[pos] > 0:
            acertos += 1
            ap += acertos / (pos + 1)
    map_k = ap / min(len(relevantes), k) if relevantes else 0.0
    return recomendados, ndcg, map_k, hit_rate


# ---------------------------------------------------------------------------
# Incerteza
# ---------------------------------------------------------------------------
def intervalo_confianca(
    valores: np.ndarray,
    confianca: float = 0.95,
    n_boot: int = 1000,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Média e IC por bootstrap (reamostragem dos usuários com reposição)."""
    valores = np.asarray(valores, dtype="float64")
    if valores.size == 0:
        return {"media": 0.0, "ic_inferior": 0.0, "ic_superior": 0.0}
    rng = np.random.default_rng(seed)
    medias = np.empty(n_boot, dtype="float64")
    for i in range(n_boot):
        amostra = rng.integers(0, valores.size, size=valores.size)
        medias[i] = valores[amostra].mean()
    alfa = (1.0 - confianca) / 2.0
    return {
        "media": float(valores.mean()),
        "ic_inferior": float(np.quantile(medias, alfa)),
        "ic_superior": float(np.quantile(medias, 1.0 - alfa)),
    }


# ---------------------------------------------------------------------------
# Métricas de catálogo
# ---------------------------------------------------------------------------
def _metricas_catalogo(
    recomendacoes: list[np.ndarray],
    n_itens: int,
    popularidade_treino: np.ndarray,
    categorias_por_item: np.ndarray | None,
    k: int,
) -> dict[str, float]:
    """Cobertura, novidade e diversidade de categoria das listas recomendadas."""
    total_visitas = float(popularidade_treino.sum()) or 1.0
    prob = np.maximum(popularidade_treino, 1.0) / total_visitas

    itens_distintos: set[int] = set()
    novidades: list[float] = []
    diversidades: list[float] = []
    for lista in recomendacoes:
        if lista.size == 0:
            continue
        itens_distintos.update(lista.tolist())
        novidades.append(float(np.mean(-np.log2(prob[lista]))))
        if categorias_por_item is not None:
            categorias = np.unique(categorias_por_item[lista])
            diversidades.append(categorias.size / min(k, np.unique(categorias_por_item).size))

    return {
        "cobertura_catalogo": len(itens_distintos) / n_itens if n_itens else 0.0,
        "novidade": float(np.mean(novidades)) if novidades else 0.0,
        "diversidade_categoria": float(np.mean(diversidades)) if diversidades else 0.0,
    }


# ---------------------------------------------------------------------------
# Baselines escaláveis (scorers)
# ---------------------------------------------------------------------------
def scorer_constante(vetor: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Scorer que devolve o mesmo vetor (popularidade, recência) a cada usuário."""

    def _scorer(usuarios: np.ndarray) -> np.ndarray:
        return np.tile(np.asarray(vetor, dtype="float64"), (len(usuarios), 1))

    return _scorer


def treinar_item_knn(
    matriz_treino: sparse.csr_matrix,
    top_vizinhos: int = TOP_VIZINHOS_KNN,
    min_similaridade: float = 1e-9,
) -> sparse.csr_matrix:
    """
    Matriz de similaridade item-item (cosseno da co-ocorrência), truncada em
    ``top_vizinhos`` por item — o que torna o método viável em catálogos grandes.
    """
    produtos = (matriz_treino > 0).astype("float64")
    coocorrencia = (produtos.T @ produtos).tocoo()
    contagens = np.asarray(produtos.sum(axis=0)).ravel()
    norma = np.sqrt(np.maximum(contagens, 1.0))
    valores = coocorrencia.data / (norma[coocorrencia.row] * norma[coocorrencia.col])
    mascara_positiva = (coocorrencia.row != coocorrencia.col) & (
        valores >= min_similaridade
    )
    linhas = coocorrencia.row[mascara_positiva]
    colunas = coocorrencia.col[mascara_positiva]
    valores = valores[mascara_positiva]

    ordem = np.lexsort((-valores, linhas))
    linhas_ord = linhas[ordem]
    _, inicio, contagens_por_linha = np.unique(
        linhas_ord, return_index=True, return_counts=True
    )
    posicao_na_linha = np.arange(linhas_ord.size) - np.repeat(inicio, contagens_por_linha)
    manter = ordem[posicao_na_linha < top_vizinhos]
    n_itens = matriz_treino.shape[1]
    return sparse.csr_matrix(
        (valores[manter], (linhas[manter], colunas[manter])),
        shape=(n_itens, n_itens),
    )


def scorer_item_knn(
    matriz_treino: sparse.csr_matrix, similaridade: sparse.csr_matrix
) -> Callable[[np.ndarray], np.ndarray]:
    """Scorer item-kNN: soma das similaridades dos itens vistos pelo usuário."""

    def _scorer(usuarios: np.ndarray) -> np.ndarray:
        historico = matriz_treino[usuarios]
        return np.asarray((historico @ similaridade).todense(), dtype="float64")

    return _scorer


def treinar_als_implicito(
    matriz_treino: sparse.csr_matrix,
    n_fatores: int = 32,
    n_iteracoes: int = 10,
    regularizacao: float = 0.05,
    alfa_confianca: float = 40.0,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ALS implícito (Hu, Koren & Volinsky, 2008) sobre feedback binário.

    Resolve, alternadamente, cada linha de usuário e de item por mínimos
    quadrados com confiança ``1 + alfa_confianca`` nos pares observados. Retorna
    ``(fatores_usuarios, fatores_itens)``. Tudo em NumPy/SciPy — sem dependência
    nova e escalável com matrizes esparsas.
    """
    rng = np.random.default_rng(seed)
    n_usuarios, n_itens = matriz_treino.shape
    x = rng.normal(0, 0.1, (n_usuarios, n_fatores))
    y = rng.normal(0, 0.1, (n_itens, n_fatores))
    conf_matrix = matriz_treino.tocsr()
    identidade = np.eye(n_fatores) * regularizacao

    for _ in range(n_iteracoes):
        x = _resolver_als(conf_matrix, y, identidade, alfa_confianca, n_fatores)
        y = _resolver_als(conf_matrix.T.tocsr(), x, identidade, alfa_confianca, n_fatores)
    return x, y


def _resolver_als(
    conf: sparse.csr_matrix,
    opostos: np.ndarray,
    identidade: np.ndarray,
    alfa: float,
    n_fatores: int,
) -> np.ndarray:
    """Um passo do ALS: resolve os fatores de um lado dados os do outro."""
    gram = opostos.T @ opostos
    resultado = np.zeros((conf.shape[0], n_fatores), dtype="float64")
    for linha in range(conf.shape[0]):
        inicio, fim = conf.indptr[linha], conf.indptr[linha + 1]
        itens = conf.indices[inicio:fim]
        if itens.size == 0:
            continue
        fatores = opostos[itens]
        a = gram + identidade + alfa * (fatores.T @ fatores)
        b = (1.0 + alfa) * fatores.sum(axis=0)
        resultado[linha] = np.linalg.solve(a, b)
    return resultado


def scorer_fatores(
    fatores_usuarios: np.ndarray, fatores_itens: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Scorer a partir de fatores latentes (ALS): produto usuário x item."""

    def _scorer(usuarios: np.ndarray) -> np.ndarray:
        return fatores_usuarios[usuarios] @ fatores_itens.T

    return _scorer


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------
def avaliar_scorer(
    scorer: Callable[[np.ndarray], np.ndarray],
    usuarios: np.ndarray,
    vistos_por_usuario: dict[int, set],
    relevantes_por_usuario: dict[int, set],
    n_itens: int,
    popularidade_treino: np.ndarray,
    categorias_por_item: np.ndarray | None = None,
    k: int = TOP_K,
    n_boot: int = 1000,
    seed: int = RANDOM_SEED,
    tamanho_bloco: int = TAMANHO_BLOCO,
) -> dict[str, Any]:
    """
    Avalia um scorer no protocolo leave-last-out e devolve métricas agregadas,
    IC por bootstrap e métricas de catálogo.

    ``usuarios`` são os ids (linhas da matriz) elegíveis; ``vistos_por_usuario`` e
    ``relevantes_por_usuario`` são indexados por esses ids.
    """
    usuarios = np.asarray(usuarios, dtype=int)
    ndcgs = np.empty(usuarios.size, dtype="float64")
    maps = np.empty(usuarios.size, dtype="float64")
    hits = np.empty(usuarios.size, dtype="float64")
    recomendacoes: list[np.ndarray] = []

    for inicio in range(0, usuarios.size, tamanho_bloco):
        bloco = usuarios[inicio : inicio + tamanho_bloco]
        scores_bloco = np.asarray(scorer(bloco), dtype="float64")
        for pos, usuario in enumerate(bloco):
            rec, ndcg, map_k, hit = _metricas_linha(
                scores_bloco[pos],
                vistos_por_usuario.get(usuario, set()),
                relevantes_por_usuario[usuario],
                k,
            )
            ndcgs[inicio + pos] = ndcg
            maps[inicio + pos] = map_k
            hits[inicio + pos] = hit
            recomendacoes.append(rec)

    resultado: dict[str, Any] = {
        f"ndcg@{k}": float(ndcgs.mean()),
        f"map@{k}": float(maps.mean()),
        f"hit_rate@{k}": float(hits.mean()),
        f"ndcg@{k}_ic": intervalo_confianca(ndcgs, n_boot=n_boot, seed=seed),
        f"map@{k}_ic": intervalo_confianca(maps, n_boot=n_boot, seed=seed),
        "num_usuarios": int(usuarios.size),
    }
    resultado.update(
        _metricas_catalogo(recomendacoes, n_itens, popularidade_treino, categorias_por_item, k)
    )
    return resultado


def matrizes_do_split(
    train_pairs: Any,
    eval_pairs: Any,
    n_usuarios: int,
    n_itens: int,
) -> tuple[sparse.csr_matrix, dict[int, set], dict[int, set]]:
    """
    Constrói a matriz binária de treino e os dicionários de vistos/relevantes.

    Espera DataFrames com colunas ``c`` (usuário) e ``p`` (item) já codificados.
    """
    linhas = train_pairs["c"].to_numpy()
    colunas = train_pairs["p"].to_numpy()
    dados = np.ones(linhas.size, dtype="float64")
    matriz = sparse.csr_matrix((dados, (linhas, colunas)), shape=(n_usuarios, n_itens))
    matriz.data[:] = 1.0
    vistos: dict[int, set] = {}
    for usuario, grupo in train_pairs.groupby("c")["p"]:
        vistos[int(usuario)] = set(grupo.tolist())
    relevantes: dict[int, set] = {}
    for usuario, grupo in eval_pairs.groupby("c")["p"]:
        relevantes[int(usuario)] = set(grupo.tolist())
    return matriz, vistos, relevantes


def usuarios_elegiveis(
    relevancia_por_usuario: dict[int, set], max_usuarios: int | None = None
) -> np.ndarray:
    """Usuários com pelo menos um item relevante no split de avaliação."""
    usuarios = np.array(sorted(relevancia_por_usuario.keys()), dtype=int)
    if max_usuarios is not None:
        usuarios = usuarios[:max_usuarios]
    return usuarios
