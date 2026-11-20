"""Testes para a normalização de dados no estágio de preparação."""

import pandas as pd

from src.scripts.preparar_dados import tratar_dados


def test_tratar_dados_normaliza_espacos_nao_separaveis() -> None:
    """Normaliza NBSP sem alterar o DataFrame de entrada."""
    dados = pd.DataFrame(
        {
            "Product Name": ["Logitech\u00a0P710e", "Nome comum"],
            "Postal Code": [None, 90210],
        }
    )

    tratados = tratar_dados(dados)

    assert tratados["Product Name"].tolist() == ["Logitech P710e", "Nome comum"]
    assert dados.loc[0, "Product Name"] == "Logitech\u00a0P710e"
    assert tratados["Postal Code"].tolist() == [5401, 90210]
