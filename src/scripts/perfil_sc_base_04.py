"""
Estágio de perfil de dados do pipeline DVC para a base `sc_base_04`.

A `sc_base_04` é um **log de eventos de interação** (visualização, carrinho e
compra) de uma loja multicategoria, em escala (~14 GB de CSV). Este estágio lê os
brutos em blocos (nunca carrega tudo na memória) e produz um relatório
reprodutível de **entendimento e qualidade**: volume, cobertura temporal,
cardinalidade de usuários/itens, distribuição do tipo de evento, nulos, preço,
densidade do sinal e cauda longa.

Princípios:
    - Nenhum número é estimado ou presumido: tudo é medido nos brutos.
    - Nada é amostrado: os identificadores são acumulados e deduplicados por
      inteiro, de modo que as contagens distintas são exatas.
    - A saída é determinística (sem carimbo de tempo), para que o DVC só
      detecte mudança quando a base ou o código mudarem de fato.

Uso:
    python -m src.scripts.perfil_sc_base_04
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.paths import REPORTS_DIR, SC_BASE_04_DIR

PERFIL_JSON = REPORTS_DIR / "perfil_sc_base_04.json"

# Colunas do log de eventos da `sc_base_04`.
COLUNA_TEMPO = "event_time"
COLUNA_TIPO = "event_type"
COLUNA_PRODUTO = "product_id"
COLUNA_CATEGORIA = "category_id"
COLUNA_CODIGO_CATEGORIA = "category_code"
COLUNA_MARCA = "brand"
COLUNA_PRECO = "price"
COLUNA_USUARIO = "user_id"
COLUNA_SESSAO = "user_session"

# `user_id` e `product_id` cabem em 32 bits; o par (usuário, produto) é
# codificado em um único uint64 de forma injetiva (sem colisão), permitindo
# contar pares distintos exatamente.
BITS_ID = np.uint64(32)
LIMITE_ID = np.uint64(2) ** BITS_ID

# Limiar de desbalanceamento entre as categorias de evento (maior / menor).
LIMIAR_DESBALANCEAMENTO_EVENTO = 10.0
TAMANHO_BLOCO = 1_000_000


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


def _distribuicao(contagens: np.ndarray) -> dict[str, float]:
    """Resumo de uma distribuição de contagens (interações por entidade)."""
    total = float(contagens.sum())
    n = int(contagens.size)
    return {
        "entidades": n,
        "interacoes": int(total),
        "media": _arredondar(total / n) if n else 0.0,
        "mediana": _arredondar(np.median(contagens)) if n else 0.0,
        "p90": _arredondar(np.quantile(contagens, 0.90)) if n else 0.0,
        "p99": _arredondar(np.quantile(contagens, 0.99)) if n else 0.0,
        "max": int(contagens.max()) if n else 0,
        "pct_com_uma_interacao": _arredondar((contagens == 1).mean() * 100, 4) if n else 0.0,
    }


def _pareto(contagens: np.ndarray) -> dict[str, float]:
    """Concentração das interações nos itens mais populares (cauda longa)."""
    ordenado = np.sort(contagens)[::-1]
    total = float(ordenado.sum())
    n = ordenado.size
    if n == 0 or total == 0:
        return {
            "top_1_pct_share": 0.0,
            "top_10_pct_share": 0.0,
            "itens_para_80_pct": 0,
            "pct_itens_para_80_pct": 0.0,
        }
    acumulado = np.cumsum(ordenado) / total
    itens_80 = int((acumulado < 0.80).sum() + 1)
    return {
        "top_1_pct_share": _arredondar(
            ordenado[: max(1, int(n * 0.01))].sum() / total * 100, 4
        ),
        "top_10_pct_share": _arredondar(
            ordenado[: max(1, int(n * 0.10))].sum() / total * 100, 4
        ),
        "itens_para_80_pct": itens_80,
        "pct_itens_para_80_pct": _arredondar(itens_80 / n * 100, 4),
    }


def _arquivos_brutos() -> list[Path]:
    """CSVs brutos da `sc_base_04`, em ordem determinística."""
    arquivos = sorted(SC_BASE_04_DIR.glob("*.csv"))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {SC_BASE_04_DIR}.\n"
            "Os brutos de `sc_base_04` devem ser obtidos na etapa de aquisição "
            "de dados (obtenção manual e externa, fora do repositório)."
        )
    return arquivos


def _varredura() -> dict[str, Any]:
    """Lê os brutos em blocos e acumula os agregados exatos."""
    n_eventos = 0
    eventos_por_tipo: Counter[str] = Counter()
    nulos: Counter[str] = Counter()
    marcas: set[str] = set()
    categorias: set[int] = set()
    categorias_codigo: set[str] = set()
    datas: set[str] = set()

    tempo_min: str | None = None
    tempo_max: str | None = None

    preco_min = np.inf
    preco_max = -np.inf
    preco_soma = 0.0
    preco_n = 0
    preco_nao_positivo = 0

    usuarios_blocos: list[np.ndarray] = []
    produtos_blocos: list[np.ndarray] = []
    pares_blocos: list[np.ndarray] = []
    id_excede_32 = False

    colunas = [
        COLUNA_TEMPO,
        COLUNA_TIPO,
        COLUNA_PRODUTO,
        COLUNA_CATEGORIA,
        COLUNA_CODIGO_CATEGORIA,
        COLUNA_MARCA,
        COLUNA_PRECO,
        COLUNA_USUARIO,
        COLUNA_SESSAO,
    ]

    for arquivo in _arquivos_brutos():
        print(f"  lendo {arquivo.name} ...")
        leitor = pd.read_csv(
            arquivo,
            usecols=colunas,
            dtype={
                COLUNA_PRODUTO: "int64",
                COLUNA_CATEGORIA: "int64",
                COLUNA_PRECO: "float64",
                COLUNA_USUARIO: "int64",
            },
            chunksize=TAMANHO_BLOCO,
        )
        for bloco in leitor:
            n_eventos += int(bloco.shape[0])

            eventos_por_tipo.update(bloco[COLUNA_TIPO].value_counts().to_dict())
            for coluna in colunas:
                nulos[coluna] += int(bloco[coluna].isnull().sum())

            marcas.update(bloco[COLUNA_MARCA].dropna().unique().tolist())
            categorias.update(bloco[COLUNA_CATEGORIA].dropna().unique().tolist())
            categorias_codigo.update(
                bloco[COLUNA_CODIGO_CATEGORIA].dropna().unique().tolist()
            )

            dias = bloco[COLUNA_TEMPO].str.slice(0, 10)
            datas.update(dias.dropna().unique().tolist())
            tempo_bloco_min = str(bloco[COLUNA_TEMPO].min())
            tempo_bloco_max = str(bloco[COLUNA_TEMPO].max())
            tempo_min = tempo_bloco_min if tempo_min is None else min(tempo_min, tempo_bloco_min)
            tempo_max = tempo_bloco_max if tempo_max is None else max(tempo_max, tempo_bloco_max)

            preco = bloco[COLUNA_PRECO]
            preco_valido = preco.dropna()
            if not preco_valido.empty:
                preco_min = min(preco_min, float(preco_valido.min()))
                preco_max = max(preco_max, float(preco_valido.max()))
                preco_soma += float(preco_valido.sum())
                preco_n += int(preco_valido.shape[0])
                preco_nao_positivo += int((preco_valido <= 0).sum())

            usuario = bloco[COLUNA_USUARIO].to_numpy(dtype=np.int64, copy=True)
            produto = bloco[COLUNA_PRODUTO].to_numpy(dtype=np.int64, copy=True)
            if usuario.max(initial=0) >= LIMITE_ID or produto.max(initial=0) >= LIMITE_ID:
                id_excede_32 = True
            usuarios_blocos.append(usuario.astype(np.uint64))
            produtos_blocos.append(produto.astype(np.uint64))
            pares_blocos.append(
                (usuario.astype(np.uint64) << BITS_ID) | produto.astype(np.uint64)
            )

    usuarios = np.concatenate(usuarios_blocos) if usuarios_blocos else np.array([], np.uint64)
    produtos = np.concatenate(produtos_blocos) if produtos_blocos else np.array([], np.uint64)
    pares = np.concatenate(pares_blocos) if pares_blocos else np.array([], np.uint64)

    _, cont_usuarios = np.unique(usuarios, return_counts=True)
    _, cont_produtos = np.unique(produtos, return_counts=True)
    n_pares_distintos = int(np.unique(pares).size)

    return {
        "n_eventos": n_eventos,
        "eventos_por_tipo": dict(eventos_por_tipo),
        "nulos": dict(nulos),
        "marcas": marcas,
        "categorias": categorias,
        "categorias_codigo": categorias_codigo,
        "datas": datas,
        "tempo_min": tempo_min,
        "tempo_max": tempo_max,
        "preco_min": preco_min,
        "preco_max": preco_max,
        "preco_soma": preco_soma,
        "preco_n": preco_n,
        "preco_nao_positivo": preco_nao_positivo,
        "cont_usuarios": cont_usuarios,
        "cont_produtos": cont_produtos,
        "n_pares_distintos": n_pares_distintos,
        "id_excede_32": id_excede_32,
    }


def _limitacoes(perfil: dict[str, Any]) -> list[str]:
    """Limitações derivadas dos números medidos."""
    limitacoes: list[str] = []

    limitacoes.append(
        "Feedback implícito e anonimizado: o evento `view` não registra a posição "
        "de impressão nem o que foi oferecido. Não há controle de exposição, logo "
        "qualquer correção por propensão é aproximada (propensão modelada), nunca "
        "um inverso de probabilidade exato."
    )
    limitacoes.append(
        "Ausência de interação não é rejeição comprovada; os negativos são "
        "'não observados'. A hipótese de amostragem negativa precisa ser explícita."
    )
    if perfil["preco"]["pct_nulo"] > 0:
        limitacoes.append(
            f"{perfil['preco']['pct_nulo']:.2f}% dos eventos não têm preço; o preço "
            "não pode ser atributo obrigatório no instante da recomendação."
        )
    if perfil["id_excede_32"]:
        limitacoes.append(
            "Há identificadores acima de 32 bits: a contagem exata de pares "
            "(usuário, produto) por codificação injetiva não se aplica."
        )
    limitacoes.append(
        "Duplicatas de linha integral não são contadas em 14 GB; a chave de "
        "deduplicação relevante é (usuário, item), medida aqui."
    )
    limitacoes.append(
        "A cobertura temporal é limitada aos arquivos presentes; tendências e "
        "sazonalidade fora da janela ficam invisíveis ao profiling."
    )
    return limitacoes


def montar_perfil() -> dict[str, Any]:
    """Consolida o perfil a partir da varredura em blocos."""
    dados = _varredura()
    n_eventos = dados["n_eventos"]

    por_tipo = dados["eventos_por_tipo"]
    tipos = sorted(por_tipo)
    eventos_por_tipo = {
        tipo: {
            "eventos": int(por_tipo[tipo]),
            "pct": _arredondar(por_tipo[tipo] / n_eventos * 100, 4),
        }
        for tipo in tipos
    }
    valores_tipo = [int(por_tipo[t]) for t in tipos]
    razao_tipo = (
        _arredondar(max(valores_tipo) / min(valores_tipo))
        if len(valores_tipo) > 1 and min(valores_tipo) > 0
        else None
    )

    cont_usuarios = dados["cont_usuarios"]
    cont_produtos = dados["cont_produtos"]
    n_usuarios = int(cont_usuarios.size)
    n_produtos = int(cont_produtos.size)
    n_pares = dados["n_pares_distintos"]
    esparsidade = (
        _arredondar(1.0 - n_pares / (n_usuarios * n_produtos))
        if n_usuarios and n_produtos
        else 0.0
    )

    colunas = [
        COLUNA_TEMPO,
        COLUNA_TIPO,
        COLUNA_PRODUTO,
        COLUNA_CATEGORIA,
        COLUNA_CODIGO_CATEGORIA,
        COLUNA_MARCA,
        COLUNA_PRECO,
        COLUNA_USUARIO,
        COLUNA_SESSAO,
    ]
    nulos = dados["nulos"]
    esquema = [
        {
            "coluna": coluna,
            "nulos": int(nulos.get(coluna, 0)),
            "pct_nulos": _arredondar(nulos.get(coluna, 0) / n_eventos * 100, 4),
        }
        for coluna in colunas
    ]

    dias = dados["datas"]
    preco_pct_nulo = _arredondar(nulos.get(COLUNA_PRECO, 0) / n_eventos * 100, 4)

    perfil: dict[str, Any] = {
        "base": "sc_base_04",
        "volume": {
            "eventos": n_eventos,
            "colunas": len(colunas),
        },
        "temporal": {
            "evento_minimo": dados["tempo_min"],
            "evento_maximo": dados["tempo_max"],
            "dias_distintos": len(dias),
        },
        "eventos_por_tipo": eventos_por_tipo,
        "razao_tipo_maior_menor": razao_tipo,
        "tipos_evento_desbalanceados": bool(
            razao_tipo is not None and razao_tipo > LIMIAR_DESBALANCEAMENTO_EVENTO
        ),
        "cardinalidade": {
            "usuarios": n_usuarios,
            "produtos": n_produtos,
            "categorias": len(dados["categorias"]),
            "codigos_categoria": len(dados["categorias_codigo"]),
            "marcas": len(dados["marcas"]),
        },
        "densidade": {
            "interacoes": n_eventos,
            "pares_usuario_produto_distintos": n_pares,
            "esparsidade_matriz": esparsidade,
            "interacoes_por_usuario": _distribuicao(cont_usuarios),
            "interacoes_por_produto": _distribuicao(cont_produtos),
        },
        "cauda_longa_produtos": {
            **_pareto(cont_produtos),
            "gini_produtos": _gini(cont_produtos),
            "gini_usuarios": _gini(cont_usuarios),
        },
        "preco": {
            "minimo": _arredondar(dados["preco_min"]) if dados["preco_n"] else None,
            "maximo": _arredondar(dados["preco_max"]) if dados["preco_n"] else None,
            "media": _arredondar(dados["preco_soma"] / dados["preco_n"])
            if dados["preco_n"]
            else None,
            "pct_nulo": preco_pct_nulo,
            "pct_nao_positivo": _arredondar(
                dados["preco_nao_positivo"] / dados["preco_n"] * 100, 4
            )
            if dados["preco_n"]
            else 0.0,
        },
        "esquema": esquema,
        "id_excede_32": dados["id_excede_32"],
    }
    perfil["limitacoes_conhecidas"] = _limitacoes(perfil)
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
    cauda = perfil["cauda_longa_produtos"]

    print("=" * 70)
    print("PERFIL DE DADOS — sc_base_04")
    print("=" * 70)
    print(f"Volume          : {volume['eventos']:,} eventos x {volume['colunas']} colunas")
    print(
        f"Cobertura       : {temporal['evento_minimo']} a {temporal['evento_maximo']} "
        f"({temporal['dias_distintos']} dias)"
    )
    print(
        f"Cardinalidade   : {perfil['cardinalidade']['usuarios']:,} usuários | "
        f"{perfil['cardinalidade']['produtos']:,} produtos | "
        f"{perfil['cardinalidade']['marcas']:,} marcas"
    )
    print(
        f"Densidade       : {densidade['pares_usuario_produto_distintos']:,} pares | "
        f"esparsidade {densidade['esparsidade_matriz']:.6f} | "
        f"mediana {densidade['interacoes_por_usuario']['mediana']:.0f} eventos/usuário"
    )
    print(
        f"Cauda longa     : top 10% dos produtos concentram "
        f"{cauda['top_10_pct_share']:.2f}% das interações "
        f"(Gini {cauda['gini_produtos']:.4f})"
    )
    print("-" * 70)
    print("Limitações conhecidas:")
    for item in perfil["limitacoes_conhecidas"]:
        print(f"  - {item}")
    print("=" * 70)
    print(f"Relatório salvo em: {PERFIL_JSON}")


def main() -> None:
    perfil = montar_perfil()
    salvar_perfil(perfil)
    imprimir_resumo(perfil)


if __name__ == "__main__":
    main()
