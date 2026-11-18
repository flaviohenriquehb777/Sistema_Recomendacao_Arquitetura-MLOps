import pandas as pd

# DOWNCAST

def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza o downcast dos tipos de dados numéricos de um DataFrame,
    reduzindo o consumo de memória sem introduzir valores nulos.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada.

    Retorna:
        pd.DataFrame: DataFrame otimizado.
    """
    df_opt = df.copy()

    # Downcast de inteiros
    for col in df_opt.select_dtypes(include=['int', 'int64', 'int32']).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast='integer')

    # Downcast de floats
    for col in df_opt.select_dtypes(include=['float', 'float64', 'float32']).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast='float')

    return df_opt


# Aplicando a função ao DataFrame transacoes

# transacoes = downcast_dataframe(transacoes)
