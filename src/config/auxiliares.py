import pandas as pd


def dataframe_coeficientes(coeficientes, colunas):
    return pd.DataFrame(
        data=coeficientes,
        index=colunas,
        columns=["coeficiente"],
    ).sort_values(by="coeficiente")