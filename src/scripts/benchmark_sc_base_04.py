"""
Benchmark de escala sobre a base **sc_base_04** (log de eventos real, ~14 GB).

Motivação
---------
A ``sc_base_04`` é o registro de interações (visualização, carrinho e compra) de
uma loja multicategoria real, com **109,95 milhões de eventos**, 5,32 milhões de
usuários e 206.876 produtos (ver ``report/perfil_sc_base_04.json``). É o cenário
em que um recomendador precisa, de fato, provar valor: catálogo grande, cauda
longa severa (Gini de produtos 0,8856) e feedback implícito.

O que este script faz
---------------------
1. Lê os brutos **uma única vez**, em blocos, mantendo apenas o necessário
   (usuário, item, instante, categoria) — nunca os 14 GB viram um DataFrame
   completo de colunas originais.
2. Aplica os filtros de tratabilidade (itens por volume, usuários por
   quantidade de itens, amostra determinística de usuários) documentados abaixo.
3. Monta o split **leave-last-out** (último evento = teste, penúltimo =
   validação) e a matriz binária de treino.
4. Treina **baselines fortes escaláveis** (sem dependência nova) e os avalia pela
   régua blindada de ``avaliacao_ranking``: NDCG@k, MAP@k e Hit-Rate@k **com
   intervalo de confiança 95% por bootstrap**, além de cobertura de catálogo,
   novidade e diversidade de categoria.

Baselines comparados
--------------------
    - aleatório (sementes fixas);
    - mais-populares (popularidade contada no treino);
    - recência (popularidade restrita à janela final);
    - item-kNN (cosseno de co-ocorrência, truncado em top-K vizinhos);
    - ALS implícito (Hu, Koren & Volinsky, 2008), resolvido em NumPy/SciPy;
    - híbrido popularity-first (popularidade + ALS), com ``alpha`` escolhido na
      validação;
    - híbrido ALS + item-kNN: prior de popularidade somado ao ALS (``alpha``) e à
      co-ocorrência (``beta``), com os dois pesos escolhidos na validação — o
      candidato de produção.

Honestidade metodológica
------------------------
    - Nenhum número é estimado ou presumido: tudo é medido nos brutos.
    - A avaliação percorre **todos** os usuários da coorte (nada é amostrado de
      forma que mude a conclusão); o bootstrap apenas quantifica a incerteza.
    - A coorte é uma **amostra determinística** de usuários elegíveis, necessária
      para tornar o ranking do catálogo completo tratável — está declarada no
      relatório de saída.
    - Feedback implícito: a ausência de interação não é rejeição comprovada; não
      há controle de exposição, logo não se calcula inverso de propensão exato.

Saídas:
    report/benchmark_sc_base_04_metricas.json

Uso:
    python -m src.scripts.benchmark_sc_base_04
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from src.config.avaliacao_ranking import (
    avaliar_scorer,
    scorer_constante,
    scorer_fatores,
    scorer_item_knn,
    treinar_als_implicito,
    treinar_item_knn,
)
from src.config.model_utils import RANDOM_SEED
from src.config.paths import REPORTS_DIR, SC_BASE_04_DIR

# ---------------------------------------------------------------------------
# Filtros de tratabilidade (declarados no relatório de saída)
# ---------------------------------------------------------------------------
# Itens com pelo menos este volume de eventos entram no catálogo candidato.
MIN_EVENTOS_ITEM = 50
# Usuários com pelo menos esta quantidade de itens distintos entram na coorte.
MIN_ITENS_USUARIO = 5
# Amostra determinística de usuários — necessária para o ranking do catálogo
# completo ser tratável em memória e tempo.
MAX_USUARIOS = 50_000

# ---------------------------------------------------------------------------
# Parâmetros dos baselines
# ---------------------------------------------------------------------------
# Corte do benchmark de escala: listas de 10 itens. O benchmark de escala mede
# os modelos em listas mais profundas que o corte k=7 do pipeline NeuMF, para
# expor de forma mais dura a separação entre personalização e popularidade.
K = 10
JANELA_RECENCIA_DIAS = 7
TOP_VIZINHOS_KNN = 100
ALS_FATORES = 32
ALS_ITERACOES = 10
N_BOOTSTRAP = 1000
USUARIOS_ALFA = 3_000
# Pesos da grade do híbrido escolhidos na validação: alpha para o ALS, beta para
# o item-kNN (o peso do prior de popularidade fica fixo em 1). ``beta=0``
# reproduz o híbrido popularity-first puro (ALS sem co-ocorrência).
ALPHAS_HIBRIDO = (0.0, 0.25, 0.5, 1.0, 2.0)
BETAS_KNN = (0.0, 0.5, 1.0, 2.0)

# Leitura em blocos (nunca carrega o arquivo inteiro de uma vez).
TAMANHO_BLOCO_LEITURA = 2_000_000

COLUNA_TEMPO = "event_time"
COLUNA_PRODUTO = "product_id"
COLUNA_CATEGORIA = "category_id"
COLUNA_USUARIO = "user_id"
COLUNAS_LIDAS = [COLUNA_TEMPO, COLUNA_PRODUTO, COLUNA_CATEGORIA, COLUNA_USUARIO]

NS_POR_DIA = 24 * 3600 * 1_000_000_000


# ---------------------------------------------------------------------------
# Leitura do log em blocos
# ---------------------------------------------------------------------------
def _arquivos_brutos() -> list[Path]:
    """CSVs brutos da ``sc_base_04``, em ordem determinística (cronológica)."""
    arquivos = sorted(SC_BASE_04_DIR.glob("*.csv"))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {SC_BASE_04_DIR}.\n"
            "Os brutos de `sc_base_04` devem ser obtidos na etapa de aquisição "
            "de dados (obtenção manual e externa, fora do repositório)."
        )
    return arquivos


def _ler_eventos() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Lê os brutos em blocos e devolve os vetores consolidados.

    Returns:
        ``(usuarios, produtos, tempos, pares_categoria)`` — os três primeiros
        alinhados por evento; ``pares_categoria`` são os pares únicos
        ``(produto, categoria)`` observados (para o mapa de categoria por item).
    """
    usuarios_b: list[np.ndarray] = []
    produtos_b: list[np.ndarray] = []
    tempos_b: list[np.ndarray] = []
    categorias_b: list[np.ndarray] = []

    for arquivo in _arquivos_brutos():
        print(f"  lendo {arquivo.name} ...", flush=True)
        leitor = pd.read_csv(
            arquivo,
            usecols=COLUNAS_LIDAS,
            dtype={COLUNA_PRODUTO: "int64", COLUNA_CATEGORIA: "int64", COLUNA_USUARIO: "int64"},
            chunksize=TAMANHO_BLOCO_LEITURA,
        )
        for bloco in leitor:
            usuarios_b.append(bloco[COLUNA_USUARIO].to_numpy(dtype="int64"))
            produtos_b.append(bloco[COLUNA_PRODUTO].to_numpy(dtype="int64"))
            tempos_b.append(
                pd.to_datetime(bloco[COLUNA_TEMPO], utc=True).astype("int64").to_numpy()
            )
            categorias_b.append(
                bloco[[COLUNA_PRODUTO, COLUNA_CATEGORIA]]
                .drop_duplicates()
                .to_numpy(dtype="int64")
            )

    usuarios = np.concatenate(usuarios_b)
    produtos = np.concatenate(produtos_b)
    tempos = np.concatenate(tempos_b)
    categorias = (
        np.concatenate(categorias_b) if categorias_b else np.empty((0, 2), dtype="int64")
    )
    return usuarios, produtos, tempos, categorias


def _mapa_categoria_por_item(categorias: np.ndarray) -> pd.Series:
    """Categoria modal por produto (determinística), a partir dos pares únicos."""
    if categorias.size == 0:
        return pd.Series(dtype="int64")
    quadros = pd.DataFrame(categorias, columns=[COLUNA_PRODUTO, COLUNA_CATEGORIA])
    return quadros.groupby(COLUNA_PRODUTO)[COLUNA_CATEGORIA].agg(
        lambda valores: valores.mode().iloc[0]
    )


# ---------------------------------------------------------------------------
# Split leave-last-out
# ---------------------------------------------------------------------------
def _evento_por_usuario(
    usuarios: np.ndarray, produtos: np.ndarray, tempos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Último e penúltimo item de cada usuário (por instante do evento).

    Returns:
        ``(usuarios_unicos, item_teste, item_validacao)`` — alinhados;
        ``item_validacao`` vale ``-1`` quando o usuário tem um único evento.
    """
    ordem = np.lexsort((tempos, usuarios))
    usuarios_ord = usuarios[ordem]
    produtos_ord = produtos[ordem]

    unicos, inicio, contagem = np.unique(
        usuarios_ord, return_index=True, return_counts=True
    )
    idx_ultimo = inicio + contagem - 1
    item_teste = produtos_ord[idx_ultimo]

    item_validacao = np.full(unicos.size, -1, dtype="int64")
    tem_anterior = contagem >= 2
    item_validacao[tem_anterior] = produtos_ord[idx_ultimo[tem_anterior] - 1]
    return unicos, item_teste, item_validacao


# ---------------------------------------------------------------------------
# Construção da coorte e das estruturas de avaliação
# ---------------------------------------------------------------------------
def _montar_coorte(
    usuarios: np.ndarray,
    produtos: np.ndarray,
    tempos: np.ndarray,
    usuarios_unicos: np.ndarray,
    item_teste: np.ndarray,
    item_validacao: np.ndarray,
) -> dict[str, Any]:
    """
    Aplica filtros, monta a matriz de treino e indexa a coorte de avaliação.

    Ordem dos filtros: item por volume → usuário por nº de itens → amostra
    determinística → exigência de que o item de teste sobreviva aos filtros.
    """
    # 1) Catálogo candidato: itens com volume mínimo (medido na base completa).
    itens_unicos, cont_item = np.unique(produtos, return_counts=True)
    mantido_item = cont_item >= MIN_EVENTOS_ITEM
    pos_global = np.searchsorted(itens_unicos, produtos)
    evento_item_mantido = mantido_item[pos_global]

    # 2) Remove o evento de teste do treino (leave-last-out).
    pos_usuario = np.searchsorted(usuarios_unicos, usuarios)
    teste_do_evento = item_teste[pos_usuario]
    evento_eh_teste = produtos == teste_do_evento
    mascara_treino = evento_item_mantido & ~evento_eh_teste

    usuarios_treino = usuarios[mascara_treino]
    itens_treino = produtos[mascara_treino]
    tempos_treino = tempos[mascara_treino]

    # 3) Estados por usuário (nº de itens no treino) e universo final de itens.
    usuarios_treino_unicos, contagem_usuario = np.unique(
        usuarios_treino, return_counts=True
    )
    itens_treino_unicos = np.unique(itens_treino)

    elegivel = contagem_usuario >= MIN_ITENS_USUARIO
    usuarios_elegiveis = usuarios_treino_unicos[elegivel]

    pos_eleg = np.searchsorted(usuarios_unicos, usuarios_elegiveis)
    teste_eleg = item_teste[pos_eleg]
    validacao_eleg = item_validacao[pos_eleg]

    # 4) O item de teste precisa existir no catálogo final (senão não há alvo).
    sobrevive = np.isin(teste_eleg, itens_treino_unicos)
    usuarios_elegiveis = usuarios_elegiveis[sobrevive]
    teste_eleg = teste_eleg[sobrevive]
    validacao_eleg = validacao_eleg[sobrevive]

    # 5) Amostra determinística de usuários (tratabilidade).
    total_elegiveis = int(usuarios_elegiveis.size)
    if usuarios_elegiveis.size > MAX_USUARIOS:
        rng = np.random.default_rng(RANDOM_SEED)
        escolha = np.sort(rng.permutation(usuarios_elegiveis.size)[:MAX_USUARIOS])
        usuarios_coorte = usuarios_elegiveis[escolha]
        teste_coorte = teste_eleg[escolha]
        validacao_coorte = validacao_eleg[escolha]
    else:
        usuarios_coorte = usuarios_elegiveis
        teste_coorte = teste_eleg
        validacao_coorte = validacao_eleg

    # 6) Matriz binária de treino restrita à coorte (reindexada).
    pos_na_coorte = np.searchsorted(usuarios_coorte, usuarios_treino)
    pertence = (pos_na_coorte < usuarios_coorte.size) & (
        usuarios_coorte[np.minimum(pos_na_coorte, usuarios_coorte.size - 1)]
        == usuarios_treino
    )
    linhas = pos_na_coorte[pertence]
    colunas = np.searchsorted(itens_treino_unicos, itens_treino[pertence])
    n_usuarios = int(usuarios_coorte.size)
    n_itens = int(itens_treino_unicos.size)
    matriz = sparse.csr_matrix(
        (np.ones(linhas.size, dtype="float64"), (linhas, colunas)),
        shape=(n_usuarios, n_itens),
    )
    matriz.data[:] = 1.0
    tempos_coorte = tempos_treino[pertence]

    # 7) Vistos (treino) e relevantes (teste/validação) por índice de coorte.
    ordem_linha = np.argsort(linhas, kind="stable")
    linhas_ord = linhas[ordem_linha]
    colunas_ord = colunas[ordem_linha]
    limites = np.searchsorted(linhas_ord, np.arange(n_usuarios + 1))
    vistos = {
        usuario: set(colunas_ord[limites[usuario] : limites[usuario + 1]].tolist())
        for usuario in range(n_usuarios)
    }
    # Item de teste é garantidamente do catálogo final (filtro `sobrevive`).
    cod_teste = np.searchsorted(itens_treino_unicos, teste_coorte)
    relevantes = {i: {int(cod_teste[i])} for i in range(n_usuarios)}

    # Validação = penúltimo item, quando ele sobrevive ao catálogo final.
    relevantes_validacao: dict[int, set[int]] = {}
    for i in range(n_usuarios):
        item_val = int(validacao_coorte[i])
        if item_val < 0:
            continue
        codigo = int(np.searchsorted(itens_treino_unicos, item_val))
        if codigo < n_itens and int(itens_treino_unicos[codigo]) == item_val:
            relevantes_validacao[i] = {codigo}

    return {
        "matriz_treino": matriz,
        "vistos": vistos,
        "relevantes": relevantes,
        "relevantes_validacao": relevantes_validacao,
        "tempos_coorte": tempos_coorte,
        "colunas": colunas,
        "n_usuarios": n_usuarios,
        "n_itens": n_itens,
        "total_elegiveis": total_elegiveis,
        "itens_catalogo": itens_treino_unicos,
    }


# ---------------------------------------------------------------------------
# Scorers e avaliação
# ---------------------------------------------------------------------------
def _popularidade(matriz: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(matriz.sum(axis=0)).ravel().astype("float64")


def _recencia(
    colunas: np.ndarray, tempos: np.ndarray, n_itens: int
) -> np.ndarray:
    """Popularidade restrita à janela final (interações mais recentes)."""
    if tempos.size == 0:
        return np.zeros(n_itens, dtype="float64")
    limite = tempos.max() - JANELA_RECENCIA_DIAS * NS_POR_DIA
    recentes = colunas[tempos >= limite]
    return np.bincount(recentes, minlength=n_itens).astype("float64")


def _zscore(vetor: np.ndarray) -> np.ndarray:
    desvio = vetor.std()
    if desvio <= 0:
        return np.zeros_like(vetor, dtype="float64")
    return (vetor - vetor.mean()) / desvio


def _scorer_hibrido(
    fatores_usuarios: np.ndarray,
    fatores_itens: np.ndarray,
    popularidade_z: np.ndarray,
    alpha: float,
):
    """Híbrido popularity-first: prior de popularidade + ajuste do ALS."""

    def _scorer(usuarios: np.ndarray) -> np.ndarray:
        pessoal = fatores_usuarios[usuarios] @ fatores_itens.T
        media = pessoal.mean(axis=1, keepdims=True)
        desvio = pessoal.std(axis=1, keepdims=True) + 1e-9
        pessoal_z = (pessoal - media) / desvio
        return popularidade_z[None, :] + alpha * pessoal_z

    return _scorer


def _zscore_linhas(matriz: np.ndarray) -> np.ndarray:
    """Padroniza cada linha (usuário) para média 0 e desvio 1."""
    media = matriz.mean(axis=1, keepdims=True)
    desvio = matriz.std(axis=1, keepdims=True) + 1e-9
    return (matriz - media) / desvio


def _scorer_hibrido_knn_als(
    matriz: sparse.csr_matrix,
    similaridade: sparse.csr_matrix,
    fatores_usuarios: np.ndarray,
    fatores_itens: np.ndarray,
    popularidade_z: np.ndarray,
    alpha: float,
    beta: float,
):
    """Híbrido combinando prior de popularidade + ALS (alpha) + item-kNN (beta)."""

    def _scorer(usuarios: np.ndarray) -> np.ndarray:
        knn = np.asarray((matriz[usuarios] @ similaridade).todense(), dtype="float64")
        als = fatores_usuarios[usuarios] @ fatores_itens.T
        return (
            popularidade_z[None, :]
            + alpha * _zscore_linhas(als)
            + beta * _zscore_linhas(knn)
        )

    return _scorer


def _avaliar(
    nome: str,
    scorer,
    dados: dict[str, Any],
    popularidade: np.ndarray,
    categorias_por_item: np.ndarray | None,
    k: int = K,
) -> dict[str, Any]:
    print(f"  avaliando {nome} ...", flush=True)
    inicio = time.time()
    resultado = avaliar_scorer(
        scorer,
        np.arange(dados["n_usuarios"]),
        dados["vistos"],
        dados["relevantes"],
        n_itens=dados["n_itens"],
        popularidade_treino=popularidade,
        categorias_por_item=categorias_por_item,
        k=k,
        n_boot=N_BOOTSTRAP,
    )
    resultado["segundos"] = round(time.time() - inicio, 2)
    print(
        f"    {nome}: NDCG@{k}={resultado[f'ndcg@{k}']:.4f} "
        f"[{resultado[f'ndcg@{k}_ic']['ic_inferior']:.4f}, "
        f"{resultado[f'ndcg@{k}_ic']['ic_superior']:.4f}]",
        flush=True,
    )
    return resultado


def _selecionar_pesos(
    matriz: sparse.csr_matrix,
    similaridade: sparse.csr_matrix,
    fatores_usuarios: np.ndarray,
    fatores_itens: np.ndarray,
    popularidade_z: np.ndarray,
    popularidade: np.ndarray,
    dados: dict[str, Any],
    k: int,
) -> tuple[float, float, dict[str, float]]:
    """
    Escolhe ``(alpha, beta)`` do híbrido pela validação (penúltimo item).

    ``alpha`` pondera o ALS e ``beta`` o item-kNN; o prior de popularidade fica
    com peso fixo em 1. A grade inclui ``beta=0``, que reproduz o híbrido
    popularity-first puro — assim o ALS com e sem co-ocorrência são comparados
    sob o mesmo critério.
    """
    relevantes_val = dados["relevantes_validacao"]
    usuarios_val = np.array(sorted(relevantes_val.keys())[:USUARIOS_ALFA], dtype=int)
    if usuarios_val.size == 0:
        return 0.0, 0.0, {}
    historico: dict[str, float] = {}
    for beta in BETAS_KNN:
        for alpha in ALPHAS_HIBRIDO:
            resultado = avaliar_scorer(
                _scorer_hibrido_knn_als(
                    matriz,
                    similaridade,
                    fatores_usuarios,
                    fatores_itens,
                    popularidade_z,
                    alpha,
                    beta,
                ),
                usuarios_val,
                dados["vistos"],
                relevantes_val,
                n_itens=dados["n_itens"],
                popularidade_treino=popularidade,
                k=k,
                n_boot=200,
            )
            historico[f"{alpha},{beta}"] = resultado[f"ndcg@{k}"]
    melhor = max(historico, key=lambda chave: historico[chave])
    alpha, beta = (float(parte) for parte in melhor.split(","))
    return alpha, beta, historico


def _melhor_alpha_sem_knn(historico: dict[str, float]) -> float:
    """Melhor ``alpha`` entre as células com ``beta = 0`` (ALS sem co-ocorrência)."""
    candidatos = {
        chave: valor for chave, valor in historico.items() if chave.endswith(",0.0")
    }
    if not candidatos:
        return 0.0
    return float(max(candidatos, key=lambda chave: candidatos[chave]).split(",")[0])


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------
def main() -> None:
    inicio_total = time.time()
    print("=" * 72)
    print("BENCHMARK DE ESCALA — sc_base_04 (baselines fortes + régua blindada)")
    print("=" * 72)

    print("\n[1/6] Lendo o log de eventos em blocos ...")
    usuarios, produtos, tempos, categorias = _ler_eventos()
    print(
        f"  {usuarios.size:,} eventos | {np.unique(usuarios).size:,} usuários | "
        f"{np.unique(produtos).size:,} produtos "
        f"({time.time() - inicio_total:.0f}s)"
    )

    print("\n[2/6] Determinando o split leave-last-out ...")
    usuarios_unicos, item_teste, item_validacao = _evento_por_usuario(
        usuarios, produtos, tempos
    )
    print(f"  {usuarios_unicos.size:,} usuários com histórico")

    print("\n[3/6] Aplicando filtros e montando a coorte ...")
    dados = _montar_coorte(
        usuarios, produtos, tempos, usuarios_unicos, item_teste, item_validacao
    )
    print(
        f"  coorte: {dados['n_usuarios']:,} usuários de "
        f"{dados['total_elegiveis']:,} elegíveis | "
        f"{dados['n_itens']:,} itens | "
        f"{dados['matriz_treino'].nnz:,} interações de treino"
    )

    matriz = dados["matriz_treino"]
    n_itens = dados["n_itens"]
    popularidade = _popularidade(matriz)
    recencia = _recencia(dados["colunas"], dados["tempos_coorte"], n_itens)

    # Categoria por item (para a métrica de diversidade).
    mapa_categoria = _mapa_categoria_por_item(categorias)
    itens_catalogo = dados["itens_catalogo"]
    categorias_por_item = (
        mapa_categoria.reindex(itens_catalogo).fillna(-1).to_numpy(dtype="int64")
        if not mapa_categoria.empty
        else None
    )

    print("\n[4/6] Treinando os baselines ...")
    similaridade = treinar_item_knn(
        matriz, top_vizinhos=TOP_VIZINHOS_KNN
    )
    print(f"  item-kNN: {similaridade.nnz:,} similaridades retidas")
    fatores_usuarios, fatores_itens = treinar_als_implicito(
        matriz, n_fatores=ALS_FATORES, n_iteracoes=ALS_ITERACOES
    )
    print(f"  ALS: fatores {fatores_usuarios.shape} x {fatores_itens.shape}")

    rng = np.random.default_rng(RANDOM_SEED)
    scorers: dict[str, Any] = {
        "aleatorio": scorer_constante(rng.random(n_itens)),
        "mais_populares": scorer_constante(popularidade),
        "recencia": scorer_constante(recencia),
        "item_knn": scorer_item_knn(matriz, similaridade),
        "als": scorer_fatores(fatores_usuarios, fatores_itens),
    }

    print("\n[5/6] Selecionando os pesos do híbrido na validação ...")
    popularidade_z = _zscore(np.log1p(popularidade))
    alpha, beta, historico_pesos = _selecionar_pesos(
        matriz,
        similaridade,
        fatores_usuarios,
        fatores_itens,
        popularidade_z,
        popularidade,
        dados,
        K,
    )
    alpha_sem_knn = _melhor_alpha_sem_knn(historico_pesos)
    if historico_pesos:
        melhores = sorted(historico_pesos.items(), key=lambda par: -par[1])[:5]
        print(
            f"  NDCG@{K} por (alpha,beta) — top 5: "
            + " | ".join(f"({chave})={valor:.4f}" for chave, valor in melhores)
        )
    print(f"  escolhido: alpha(ALS)={alpha} beta(kNN)={beta}")
    scorers["hibrido_popularity_first"] = _scorer_hibrido(
        fatores_usuarios, fatores_itens, popularidade_z, alpha_sem_knn
    )
    scorers["hibrido_knn_als"] = _scorer_hibrido_knn_als(
        matriz,
        similaridade,
        fatores_usuarios,
        fatores_itens,
        popularidade_z,
        alpha,
        beta,
    )

    print("\n[6/6] Avaliando na coorte de teste (ranking do catálogo completo) ...")
    metricas = {
        nome: _avaliar(nome, scorer, dados, popularidade, categorias_por_item)
        for nome, scorer in scorers.items()
    }

    # --- Relatório -----------------------------------------------------------
    ndcg_als = metricas["als"][f"ndcg@{K}"]
    ndcg_knn = metricas["item_knn"][f"ndcg@{K}"]
    ndcg_pop = metricas["mais_populares"][f"ndcg@{K}"]
    ndcg_hib = metricas["hibrido_knn_als"][f"ndcg@{K}"]
    melhor_baseline = max(ndcg_als, ndcg_knn, ndcg_pop)

    relatorio: dict[str, Any] = {
        "base": "sc_base_04",
        "escala": {
            "eventos": int(usuarios.size),
            "usuarios": int(usuarios_unicos.size),
            "produtos": int(np.unique(produtos).size),
        },
        "filtros": {
            "min_eventos_item": MIN_EVENTOS_ITEM,
            "min_itens_usuario": MIN_ITENS_USUARIO,
            "max_usuarios_coorte": MAX_USUARIOS,
            "usuarios_elegiveis": int(dados["total_elegiveis"]),
            "usuarios_coorte": int(dados["n_usuarios"]),
            "itens_catalogo": int(dados["n_itens"]),
        },
        "protocolo": {
            "split": "leave-last-out (último evento = teste; penúltimo = validação)",
            "candidatos": "catálogo completo, com exclusão dos itens vistos",
            "feedback": "implícito binário (todos os tipos de evento)",
            "k": K,
            "n_bootstrap": N_BOOTSTRAP,
            "semente": RANDOM_SEED,
        },
        "baselines": {
            "janela_recencia_dias": JANELA_RECENCIA_DIAS,
            "top_vizinhos_knn": TOP_VIZINHOS_KNN,
            "als_fatores": ALS_FATORES,
            "als_iteracoes": ALS_ITERACOES,
            "pesos_hibrido_escolhidos": {"alpha_als": alpha, "beta_knn": beta},
            "alpha_als_sem_knn": alpha_sem_knn,
            "historico_pesos_validacao": historico_pesos,
        },
        "metricas": metricas,
        "comparacao": {
            "melhor_baseline_ndcg": melhor_baseline,
            "ganho_hibrido_sobre_melhor_baseline": (
                ndcg_hib / melhor_baseline - 1 if melhor_baseline else None
            ),
            "ganho_als_sobre_populares": (ndcg_als / ndcg_pop - 1 if ndcg_pop else None),
            "ganho_knn_sobre_populares": (ndcg_knn / ndcg_pop - 1 if ndcg_pop else None),
        },
        "limitacoes_conhecidas": [
            (
                "A coorte é uma amostra determinística de usuários elegíveis — "
                "necessária para o ranking do catálogo completo ser tratável."
            ),
            (
                "Feedback implícito sem controle de exposição: correção por propensão "
                "é aproximada, nunca um inverso de probabilidade exato."
            ),
            "A ausência de interação é 'não observada', não rejeição comprovada.",
        ],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    caminho = REPORTS_DIR / "benchmark_sc_base_04_metricas.json"
    caminho.write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\n{'=' * 72}")
    print(f"RESUMO — NDCG@{K} (IC 95%)")
    print("=" * 72)
    for nome, valores in metricas.items():
        ic = valores[f"ndcg@{K}_ic"]
        print(
            f"  {nome:<28} {valores[f'ndcg@{K}']:.4f} "
            f"[{ic['ic_inferior']:.4f}, {ic['ic_superior']:.4f}]  "
            f"cov={valores['cobertura_catalogo']:.3f} "
            f"nov={valores['novidade']:.2f} "
            f"div={valores['diversidade_categoria']:.3f}"
        )
    print("=" * 72)
    print(f"Relatório salvo em: {caminho}")
    print(f"Tempo total: {time.time() - inicio_total:.0f}s")


if __name__ == "__main__":
    main()
