#!/usr/bin/env python3
"""
Estágio de preparação de dados do pipeline DVC.

Lê a base bruta (`sc_base_01`) em CSV, aplica o tratamento mínimo necessário
(normalização de espaços Unicode, correção de valores nulos e downcast dos tipos numéricos) e grava a base
tratada em Parquet, que é consumida pelos estágios de treinamento.

Reproduz fielmente o tratamento definido em
`notebooks/01_777_Sistema_Recomendacao_Inicial.ipynb`:
    read_csv -> preencher 'Postal Code' nulo (5401) -> downcast_dataframe -> to_parquet

Uso:
    python src/scripts/preparar_dados.py
"""

import pandas as pd

from src.config.auxiliares_ml import downcast_dataframe
from src.config.paths import DADOS_BRUTOS, DADOS_TRATADOS

# Valor aplicado pelo notebook original aos 'Postal Code' nulos.
POSTAL_CODE_PADRAO = 5401


def carregar_dados_brutos() -> pd.DataFrame:
    """Lê a base bruta em CSV."""
    return pd.read_csv(DADOS_BRUTOS)


def tratar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o tratamento mínimo da base:
      1. Substitui espaços não separáveis por espaços comuns nas colunas textuais.
      2. Preenche os valores nulos de 'Postal Code' com o CEP padrão.
      3. Faz o downcast dos tipos numéricos para reduzir o uso de memória.
    """
    df = df.copy()

    colunas_textuais = df.select_dtypes(include=["object", "string"]).columns
    for coluna in colunas_textuais:
        df[coluna] = df[coluna].str.replace("\u00a0", " ", regex=False)

    if "Postal Code" in df.columns:
        df.loc[df["Postal Code"].isnull(), "Postal Code"] = POSTAL_CODE_PADRAO

    return downcast_dataframe(df)


def salvar_dados_tratados(df: pd.DataFrame) -> None:
    """Grava a base tratada em formato Parquet."""
    DADOS_TRATADOS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DADOS_TRATADOS)


def main() -> None:
    print("=" * 70)
    print("PREPARAÇÃO DE DADOS")
    print("=" * 70)

    print(f"Lendo base bruta: {DADOS_BRUTOS}")
    df_bruto = carregar_dados_brutos()
    print(
        f"Base bruta carregada: {df_bruto.shape[0]} linhas x {df_bruto.shape[1]} colunas"
    )

    nulos_postal = (
        int(df_bruto["Postal Code"].isnull().sum())
        if "Postal Code" in df_bruto.columns
        else 0
    )
    print(f"Valores nulos em 'Postal Code': {nulos_postal}")

    df_tratado = tratar_dados(df_bruto)
    print(f"Base tratada: {df_tratado.shape[0]} linhas x {df_tratado.shape[1]} colunas")

    salvar_dados_tratados(df_tratado)
    print(f"Base tratada salva em: {DADOS_TRATADOS}")
    print("=" * 70)


if __name__ == "__main__":
    main()
