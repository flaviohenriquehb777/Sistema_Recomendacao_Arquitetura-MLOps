"""
Preparação da base pública **sc_base_03** (e-commerce real, feedback implícito).

Origem e honestidade
--------------------
A sc_base_03 é um *log de eventos* real de um site de e-commerce, publicado no
portal de dados (``sc_base_03/ecommerce-dataset``), cobrindo ~4,5 meses. É **dado real**,
não sintético — porém **anonimizado** (não há nomes de produto nem de cliente).

Eventos (``events.csv``): ``timestamp, visitorid, event, itemid, transactionid``,
com ``event`` ∈ {``view``, ``addtocart``, ``transaction``}. Tratamos o conjunto
de eventos como **feedback implícito binário** (interagiu / não interagiu).

Por que esta base?
    A base própria (sc_base_01) é pequena e esparsa demais para um modelo
    neural superar a popularidade. A sc_base_03 traz **sinal implícito real**
    (e dezenas de milhares de interações úteis) — o cenário em que um
    recomendador de fato demonstra valor.

Tratabilidade (filtros explícitos, aplicados nesta ordem)
---------------------------------------------------------
1. **Itens**: mantém itens com pelo menos ``min_eventos_item`` eventos (medido na
   base completa) — remove a cauda longa de itens vistos 1–2 vezes.
2. **Usuários**: mantém visitantes com pelo menos ``min_itens_usuario`` itens
   distintos — remove sessões "bounce" (a mediana da base é 1 item/visitante).
3. **Amostra**: se ainda houver mais de ``max_clientes`` visitantes, amostra
   determinística (semente fixa) para tornar a avaliação com *ranking do
   catálogo completo* viável.

Metadados de item: ``Category`` vem da propriedade ``categoryid`` de
``item_properties_*.csv`` (valor mais recente por item) e ``Sub-Category`` vem do
``parentid`` da ``category_tree.csv``. Itens sem categoria recebem o rótulo
sentinela ``sem_categoria``.

O esquema de saída é o **mesmo** consumido por todo o pipeline de recomendação:
``Customer Name, Product ID, Product Name, Category, Sub-Category, Order Date``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import DATA_DIR

# Diretório com os arquivos brutos baixados do portal de dados (ver README).
sc_base_03_RAW_DIR = DATA_DIR / "sc_base_03"

# Esquema padrão consumido pelo pipeline de recomendação.
COLUNAS_PADRAO = [
    "Customer Name",
    "Product ID",
    "Product Name",
    "Category",
    "Sub-Category",
    "Order Date",
]

CATEGORIA_DESCONHECIDA = "sem_categoria"
PROPRIEDADE_CATEGORIA = "categoryid"
CHUNK_PROPRIEDADES = 2_000_000


def carregar_eventos(dir_bruto: Path = sc_base_03_RAW_DIR) -> pd.DataFrame:
    """Carrega ``events.csv`` e converte o timestamp (ms epoch) em datetime."""
    caminho = Path(dir_bruto) / "events.csv"
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}.\n"
            "Baixe a base sc_base_03 do portal de dados e extraia em 'dados/sc_base_03':\n"
            "    portal de dados datasets download -d sc_base_03/ecommerce-dataset "
            "-p dados/sc_base_03 --unzip"
        )
    eventos = pd.read_csv(caminho)
    eventos["timestamp"] = pd.to_datetime(eventos["timestamp"], unit="ms")
    return eventos[["timestamp", "visitorid", "event", "itemid"]]


def carregar_categorias(dir_bruto: Path = sc_base_03_RAW_DIR) -> pd.Series:
    """
    Mapeia ``itemid`` -> ``categoryid`` (valor mais recente por item).

    Lê ``item_properties_part*.csv`` em blocos e mantém apenas as linhas cuja
    propriedade é ``categoryid``, para não carregar ~900 MB de propriedades
    irrelevantes na memória.
    """
    blocos = []
    for parte in (1, 2):
        caminho = Path(dir_bruto) / f"item_properties_part{parte}.csv"
        if not caminho.exists():
            continue
        for chunk in pd.read_csv(caminho, chunksize=CHUNK_PROPRIEDADES):
            sub = chunk.loc[
                chunk["property"] == PROPRIEDADE_CATEGORIA, ["timestamp", "itemid", "value"]
            ]
            if not sub.empty:
                blocos.append(sub)
    if not blocos:
        return pd.Series(dtype="object", name=PROPRIEDADE_CATEGORIA)
    categorias = pd.concat(blocos, ignore_index=True).sort_values("timestamp")
    categorias = categorias.drop_duplicates(subset="itemid", keep="last")
    return categorias.set_index("itemid")["value"].astype(str).rename(PROPRIEDADE_CATEGORIA)


def carregar_arvore_categorias(dir_bruto: Path = sc_base_03_RAW_DIR) -> pd.Series:
    """Mapeia ``categoryid`` -> ``parentid`` a partir de ``category_tree.csv``."""
    caminho = Path(dir_bruto) / "category_tree.csv"
    if not caminho.exists():
        return pd.Series(dtype="object", name="parentid")
    arvore = pd.read_csv(caminho)
    return arvore.set_index("categoryid")["parentid"].rename("parentid")


def construir_interacoes(
    dir_bruto: Path = sc_base_03_RAW_DIR,
    min_eventos_item: int = 100,
    min_itens_usuario: int = 5,
    max_clientes: int | None = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Constrói o DataFrame de interações (esquema padrão) a partir da base bruta.

    Aplica os filtros de tratabilidade descritos no docstring do módulo e agrega
    os metadados de categoria/subcategoria. Interações repetidas do mesmo
    (cliente, item) permanecem com seus timestamps originais — o *dedup* e o
    split *leave-last-out* ficam a cargo de ``model_utils``.
    """
    eventos = carregar_eventos(dir_bruto)

    # 1) Itens relevantes (medido na base completa).
    eventos_itens = eventos.groupby("itemid")["visitorid"].size()
    itens_relevantes = eventos_itens.index[eventos_itens >= min_eventos_item]
    eventos = eventos.loc[eventos["itemid"].isin(itens_relevantes)]

    # 2) Usuários engajados.
    itens_por_usuario = eventos.groupby("visitorid")["itemid"].nunique()
    usuarios_relevantes = itens_por_usuario.index[itens_por_usuario >= min_itens_usuario]
    eventos = eventos.loc[eventos["visitorid"].isin(usuarios_relevantes)]

    # 3) Amostra determinística de clientes (tratabilidade da avaliação).
    if max_clientes is not None:
        usuarios = np.sort(eventos["visitorid"].unique())
        if usuarios.size > max_clientes:
            rng = np.random.default_rng(seed)
            escolhidos = rng.choice(usuarios, size=max_clientes, replace=False)
            eventos = eventos.loc[eventos["visitorid"].isin(escolhidos)]

    # Metadados de categoria (item -> categoryid -> parentid).
    categorias = carregar_categorias(dir_bruto)
    pais = carregar_arvore_categorias(dir_bruto)
    categoria_item = eventos["itemid"].map(categorias)
    subcategoria_item = categoria_item.map(pais).fillna(categoria_item)

    interacoes = pd.DataFrame(
        {
            "Customer Name": "cliente_" + eventos["visitorid"].astype(str),
            "Product ID": "item_" + eventos["itemid"].astype(str),
            "Product Name": "item_" + eventos["itemid"].astype(str),
            "Category": categoria_item.fillna(CATEGORIA_DESCONHECIDA).astype(str),
            "Sub-Category": subcategoria_item.fillna(CATEGORIA_DESCONHECIDA).astype(str),
            "Order Date": eventos["timestamp"],
        }
    )
    return (
        interacoes.sort_values(["Customer Name", "Order Date"])
        .reset_index(drop=True)[COLUNAS_PADRAO]
    )


def resumo_base(df: pd.DataFrame) -> dict:
    """Estatísticas descritivas da base (para o relatório de benchmark)."""
    clientes = int(df["Customer Name"].nunique())
    produtos = int(df["Product ID"].nunique())
    interacoes = int(len(df))
    por_cliente = df.groupby("Customer Name")["Product ID"].nunique()
    densidade = interacoes / (clientes * produtos) if clientes and produtos else float("nan")
    return {
        "interacoes": interacoes,
        "clientes": clientes,
        "produtos": produtos,
        "categorias": int(df["Category"].nunique()),
        "subcategorias": int(df["Sub-Category"].nunique()),
        "itens_por_cliente_media": float(por_cliente.mean()),
        "itens_por_cliente_mediana": float(por_cliente.median()),
        "densidade": float(densidade),
    }
