"""
Estágio de perfil de dados do pipeline DVC (Etapa 1 do roadmap).

Lê a base bruta `sc_base_01` e produz um relatório reprodutível de
**entendimento e qualidade** dos dados: volume, esquema, nulos, duplicatas,
cobertura temporal, densidade do sinal de interação, cauda longa,
representatividade das categorias e limitações conhecidas.

Princípios:
    - Nenhum número é estimado ou presumido: tudo é medido na base bruta.
    - A saída é determinística (sem carimbo de tempo), para que o DVC só
      detecte mudança quando a base ou o código mudarem de fato.

Uso:
    python -m src.scripts.perfil_dados
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.config.paths import DADOS_BRUTOS, REPORTS_DIR, exigir_dados_brutos

COLUNA_PEDIDO = "Order ID"
COLUNA_CLIENTE = "Customer ID"
COLUNA_PRODUTO = "Product ID"
COLUNA_DATA_PEDIDO = "Order Date"
COLUNA_DATA_ENVIO = "Ship Date"
COLUNA_VALOR = "Sales"
FORMATO_DATA = "%d/%m/%Y"

# Limiar de desbalanceamento entre categorias (razão entre a maior e a menor).
LIMIAR_DESBALANCEAMENTO = 10.0
# Limiar de esparsidade que caracteriza "densidade baixa".
LIMIAR_ESPARSIDADE_ALTA = 0.99
# Só expõe o valor modal de uma coluna quando a cardinalidade é baixa o bastante
# para o valor ser uma categoria — nunca um registro individual (evita PII).
LIMIAR_EXEMPLO_CARDINALIDADE = 50

PERFIL_JSON = REPORTS_DIR / "perfil_sc_base_01.json"


def carregar_base() -> pd.DataFrame:
    """Lê a base bruta em CSV."""
    return pd.read_csv(exigir_dados_brutos())


def _arredondar(valor: float, casas: int = 6) -> float:
    """Arredonda para manter o JSON estável entre execuções."""
    return float(round(float(valor), casas))


def _gini(valores: np.ndarray) -> float:
    """Coeficiente de Gini de uma distribuição não negativa (0 = uniforme)."""
    x = np.sort(np.asarray(valores, dtype=float))
    n = x.size
    total = x.sum()
    if n == 0 or total == 0:
        return 0.0
    indices = np.arange(1, n + 1)
    return _arredondar((2.0 * np.sum(indices * x)) / (n * total) - (n + 1) / n)


def perfil_volume(df: pd.DataFrame) -> dict[str, Any]:
    """Volume e forma da base."""
    return {
        "linhas": int(df.shape[0]),
        "colunas": int(df.shape[1]),
        "memoria_mb": _arredondar(df.memory_usage(deep=True).sum() / 1024**2, 3),
    }


def perfil_esquema(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Esquema coluna a coluna: tipo, nulos, cardinalidade e valor modal seguro.

    O valor modal só é exposto em colunas de baixa cardinalidade (categorias).
    Colunas de identificação (IDs, nomes, cidades) nunca têm amostra publicada.
    """
    colunas: list[dict[str, Any]] = []
    for nome in df.columns:
        serie = df[nome]
        n_nulos = int(serie.isnull().sum())
        n_unicos = int(serie.nunique(dropna=True))
        sem_nulos = serie.dropna()
        valor_modal = None
        if 0 < n_unicos <= LIMIAR_EXEMPLO_CARDINALIDADE and not sem_nulos.empty:
            valor_modal = str(sem_nulos.mode().iloc[0])[:80]
        colunas.append(
            {
                "coluna": str(nome),
                "tipo": str(serie.dtype),
                "nulos": n_nulos,
                "pct_nulos": _arredondar(n_nulos / len(df) * 100, 4),
                "unicos": n_unicos,
                "valor_modal": valor_modal,
            }
        )
    return colunas


def perfil_duplicatas(df: pd.DataFrame) -> dict[str, Any]:
    """Duplicatas de linha integral e da chave de negócio (pedido, produto)."""
    chave = [COLUNA_PEDIDO, COLUNA_PRODUTO]
    n_pedidos = int(df[COLUNA_PEDIDO].nunique())
    itens_por_pedido = df.groupby(COLUNA_PEDIDO).size()
    return {
        "linhas_duplicadas_integrais": int(df.duplicated().sum()),
        "chave_pedido_produto_repetida": int(df.duplicated(subset=chave).sum()),
        "pedidos_distintos": n_pedidos,
        "itens_por_pedido_media": _arredondar(itens_por_pedido.mean()),
        "itens_por_pedido_max": int(itens_por_pedido.max()),
    }


def perfil_temporal(df: pd.DataFrame) -> dict[str, Any]:
    """Cobertura temporal e invariantes das datas do evento."""
    data_pedido = pd.to_datetime(df[COLUNA_DATA_PEDIDO], format=FORMATO_DATA)
    data_envio = pd.to_datetime(df[COLUNA_DATA_ENVIO], format=FORMATO_DATA)

    falhas_parsing = int(data_pedido.isnull().sum() + data_envio.isnull().sum())
    atraso = (data_envio - data_pedido).dt.days
    meses = data_pedido.dt.to_period("M").nunique()

    return {
        "data_minima": data_pedido.min().strftime("%Y-%m-%d"),
        "data_maxima": data_pedido.max().strftime("%Y-%m-%d"),
        "amplitude_dias": int((data_pedido.max() - data_pedido.min()).days),
        "meses_cobertos": int(meses),
        "dias_distintos_com_pedido": int(data_pedido.dt.normalize().nunique()),
        "falhas_de_parsing": falhas_parsing,
        "remessas_antes_do_pedido": int((atraso < 0).sum()),
        "prazo_envio_dias_medio": _arredondar(atraso.mean()),
        "prazo_envio_dias_p95": _arredondar(atraso.quantile(0.95)),
    }


def perfil_densidade(df: pd.DataFrame) -> dict[str, Any]:
    """Densidade do sinal de interação (cliente x produto)."""
    por_cliente = df.groupby(COLUNA_CLIENTE).size()
    por_produto = df.groupby(COLUNA_PRODUTO).size()

    n_clientes = int(por_cliente.size)
    n_produtos = int(por_produto.size)
    n_pares = int(df.drop_duplicates([COLUNA_CLIENTE, COLUNA_PRODUTO]).shape[0])

    return {
        "clientes_distintos": n_clientes,
        "produtos_distintos": n_produtos,
        "interacoes": int(df.shape[0]),
        "pares_cliente_produto_distintos": n_pares,
        "interacoes_por_cliente": {
            "media": _arredondar(por_cliente.mean()),
            "mediana": _arredondar(por_cliente.median()),
            "p25": _arredondar(por_cliente.quantile(0.25)),
            "p75": _arredondar(por_cliente.quantile(0.75)),
            "max": int(por_cliente.max()),
        },
        "interacoes_por_produto": {
            "media": _arredondar(por_produto.mean()),
            "mediana": _arredondar(por_produto.median()),
            "max": int(por_produto.max()),
        },
        "pct_clientes_com_uma_interacao": _arredondar(
            (por_cliente == 1).mean() * 100, 4
        ),
        "pct_produtos_com_uma_interacao": _arredondar(
            (por_produto == 1).mean() * 100, 4
        ),
        "esparsidade_matriz": _arredondar(1.0 - n_pares / (n_clientes * n_produtos)),
    }


def perfil_cauda_longa(df: pd.DataFrame) -> dict[str, Any]:
    """Concentração das interações: cauda longa de produtos e de clientes."""
    volumes_produto = df.groupby(COLUNA_PRODUTO).size().sort_values(ascending=False)
    volumes_cliente = df.groupby(COLUNA_CLIENTE).size().sort_values(ascending=False)

    total = float(volumes_produto.sum())
    acumulado = volumes_produto.cumsum() / total
    n_produtos_pareto = int((acumulado < 0.80).sum() + 1)

    return {
        "top_1_pct_produtos_share": _arredondar(
            volumes_produto.head(max(1, int(len(volumes_produto) * 0.01))).sum()
            / total
            * 100,
            4,
        ),
        "top_10_pct_produtos_share": _arredondar(
            volumes_produto.head(max(1, int(len(volumes_produto) * 0.10))).sum()
            / total
            * 100,
            4,
        ),
        "produtos_para_80_pct_interacoes": n_produtos_pareto,
        "pct_produtos_para_80_pct_interacoes": _arredondar(
            n_produtos_pareto / len(volumes_produto) * 100, 4
        ),
        "gini_produtos": _gini(volumes_produto.to_numpy()),
        "gini_clientes": _gini(volumes_cliente.to_numpy()),
    }


def perfil_representatividade(df: pd.DataFrame) -> dict[str, Any]:
    """Distribuição das dimensões categóricas e alerta de desbalanceamento."""
    resultado: dict[str, Any] = {}
    for coluna in ["Segment", "Region", "Category", "Sub-Category"]:
        if coluna not in df.columns:
            continue
        distribuicao = df[coluna].value_counts(normalize=True) * 100
        razao = (
            _arredondar(distribuicao.max() / distribuicao.min())
            if distribuicao.size > 1 and distribuicao.min() > 0
            else None
        )
        resultado[coluna] = {
            "valores_distintos": int(distribuicao.size),
            "distribuicao_pct": {
                str(k): _arredondar(v, 4) for k, v in distribuicao.items()
            },
            "razao_maior_menor": razao,
            "desbalanceada": bool(razao is not None and razao > LIMIAR_DESBALANCEAMENTO),
        }
    return resultado


def limitacoes_conhecidas(perfil: dict[str, Any]) -> list[str]:
    """Limitações derivadas dos números medidos (nada é afirmado sem respaldo)."""
    densidade = perfil["densidade"]
    temporal = perfil["temporal"]
    duplicatas = perfil["duplicatas"]
    limitacoes: list[str] = []

    if densidade["esparsidade_matriz"] >= LIMIAR_ESPARSIDADE_ALTA:
        limitacoes.append(
            "Matriz cliente x produto com esparsidade de "
            f"{densidade['esparsidade_matriz']:.4f} ({densidade['pares_cliente_produto_distintos']} "
            f"pares observados para {densidade['clientes_distintos']} clientes x "
            f"{densidade['produtos_distintos']} produtos): qualquer métrica de ranking "
            "tem alta variância e deve ser lida com intervalo de confiança."
        )

    limitacoes.append(
        "Sinal de interação exclusivamente positivo (compra registrada). Não há "
        "evento de impressão/exposição, portanto não é possível distinguir "
        "'não gostou' de 'não viu' — o modelo é avaliado por relevância observada, "
        "nunca por taxa de clique."
    )

    if temporal["remessas_antes_do_pedido"] > 0:
        limitacoes.append(
            f"{temporal['remessas_antes_do_pedido']} registros têm Ship Date anterior "
            "a Order Date (inconsistência a tratar na preparação)."
        )
    limitacoes.append(
        "Ship Date é posterior ao evento de compra: não pode ser usada como "
        "atributo de entrada sob um protocolo temporal honesto."
    )

    if duplicatas["chave_pedido_produto_repetida"] > 0:
        limitacoes.append(
            f"{duplicatas['chave_pedido_produto_repetida']} linhas repetem a chave "
            "(Order ID, Product ID): a agregação precisa ser definida por cliente-item."
        )

    limitacoes.append(
        "A granularidade da linha é item de pedido: itens comprados no mesmo pedido "
        "co-ocorrem por decisão de compra, o que infla a similaridade item-item se "
        "ignorada a estrutura do pedido."
    )
    return limitacoes


def montar_perfil(df: pd.DataFrame) -> dict[str, Any]:
    """Consolida todas as seções do perfil em um único documento."""
    perfil: dict[str, Any] = {
        "base": DADOS_BRUTOS.name,
        "volume": perfil_volume(df),
        "temporal": perfil_temporal(df),
        "duplicatas": perfil_duplicatas(df),
        "densidade": perfil_densidade(df),
        "cauda_longa": perfil_cauda_longa(df),
        "representatividade": perfil_representatividade(df),
        "esquema": perfil_esquema(df),
    }
    perfil["limitacoes_conhecidas"] = limitacoes_conhecidas(perfil)
    return perfil


def salvar_perfil(perfil: dict[str, Any]) -> None:
    """Grava o perfil em JSON (determinístico)."""
    PERFIL_JSON.parent.mkdir(parents=True, exist_ok=True)
    PERFIL_JSON.write_text(
        json.dumps(perfil, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def imprimir_resumo(perfil: dict[str, Any]) -> None:
    """Resumo legível do perfil no terminal."""
    volume = perfil["volume"]
    temporal = perfil["temporal"]
    densidade = perfil["densidade"]
    cauda = perfil["cauda_longa"]

    print("=" * 70)
    print("PERFIL DE DADOS — sc_base_01")
    print("=" * 70)
    print(f"Volume          : {volume['linhas']} linhas x {volume['colunas']} colunas")
    print(
        f"Cobertura       : {temporal['data_minima']} a {temporal['data_maxima']} "
        f"({temporal['meses_cobertos']} meses)"
    )
    print(
        f"Interações      : {densidade['interacoes']} | "
        f"{densidade['clientes_distintos']} clientes | "
        f"{densidade['produtos_distintos']} produtos"
    )
    print(
        f"Densidade       : esparsidade {densidade['esparsidade_matriz']:.4f} | "
        f"mediana de {densidade['interacoes_por_cliente']['mediana']:.0f} itens/cliente"
    )
    print(
        f"Cauda longa     : top 10% dos produtos concentram "
        f"{cauda['top_10_pct_produtos_share']:.2f}% das interações "
        f"(Gini {cauda['gini_produtos']:.4f})"
    )
    print("-" * 70)
    print("Limitações conhecidas:")
    for item in perfil["limitacoes_conhecidas"]:
        print(f"  - {item}")
    print("=" * 70)
    print(f"Relatório salvo em: {PERFIL_JSON}")


def main() -> None:
    print(f"Lendo base bruta: {DADOS_BRUTOS}")
    df = carregar_base()
    perfil = montar_perfil(df)
    salvar_perfil(perfil)
    imprimir_resumo(perfil)


if __name__ == "__main__":
    main()
