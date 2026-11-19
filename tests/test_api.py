"""
Testes da API de inferência (``serving/app.py``) usando o ``TestClient`` do FastAPI.

Usa um recomendador SINTÉTICO pequeno, construído em memória, para não depender
dos artefatos versionados no DVC nem da base real — assim os testes são rápidos
e reprodutíveis em qualquer ambiente, inclusive no CI (onde `models/` e
`dados_tratados.parquet` não estão no git).
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import LabelEncoder

from serving.app import create_app
from serving.service import RecommenderService
from src.config.model_utils import (
    build_item_catalog,
    compute_popularity_prior,
    create_ranking_model,
)

NUM_CUSTOMERS, NUM_PRODUCTS, NUM_CATEGORIES, NUM_SUBCATEGORIES = 3, 6, 3, 3
SHAP_FEATURES = {"cliente", "produto", "categoria", "subcategoria"}


def _dados_sinteticos() -> pd.DataFrame:
    """Base mínima: 3 clientes, 6 produtos, 2 itens vistos por cliente."""
    return pd.DataFrame(
        {
            "Customer Name": ["Ana", "Ana", "Bruno", "Bruno", "Carla", "Carla"],
            "Product ID": ["P1", "P2", "P3", "P4", "P5", "P6"],
            "Product Name": [f"Produto {i}" for i in range(1, 7)],
            "Category": ["Cat A", "Cat A", "Cat B", "Cat B", "Cat C", "Cat C"],
            "Sub-Category": ["Sub 1", "Sub 1", "Sub 2", "Sub 2", "Sub 3", "Sub 3"],
        }
    )


@pytest.fixture(scope="module")
def service() -> RecommenderService:
    """Recomendador sintético (modelo NeuMF mínimo + encoders + catálogo)."""
    df = _dados_sinteticos()
    encoders = {
        "customer": LabelEncoder().fit(df["Customer Name"]),
        "product": LabelEncoder().fit(df["Product ID"]),
        "category": LabelEncoder().fit(df["Category"]),
        "subcategory": LabelEncoder().fit(df["Sub-Category"]),
    }
    catalog = build_item_catalog(
        df, encoders["product"], encoders["category"], encoders["subcategory"]
    )
    model = create_ranking_model(
        NUM_CUSTOMERS, NUM_PRODUCTS, NUM_CATEGORIES, NUM_SUBCATEGORIES, embedding_dim=4
    )
    train_pairs = pd.DataFrame(
        {
            "c": encoders["customer"].transform(df["Customer Name"]),
            "p": encoders["product"].transform(df["Product ID"]),
        }
    )
    popularity = compute_popularity_prior(train_pairs, NUM_PRODUCTS)
    return RecommenderService(
        model=model,
        df=df,
        encoders=encoders,
        catalog=catalog,
        alpha=0.5,
        popularity=popularity,
        background_size=6,
        shap_nsamples=50,
    )


@pytest.fixture(scope="module")
def client(service: RecommenderService):
    with TestClient(create_app(service=service)) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    resposta = client.get("/health")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["status"] == "ok"
    assert corpo["model_loaded"] is True
    assert corpo["num_products"] == NUM_PRODUCTS
    assert corpo["num_customers"] == NUM_CUSTOMERS


def test_recommend_cliente_valido_com_shap(client: TestClient) -> None:
    resposta = client.post("/recommend", json={"customer_name": "Ana", "top_k": 4})
    assert resposta.status_code == 200
    corpo = resposta.json()

    assert corpo["cold_start"] is False
    assert corpo["top_k"] == 4
    assert len(corpo["recommendations"]) == 4  # 6 produtos - 2 vistos = 4 candidatos
    assert corpo["base_value"] is not None

    # Ranking exclui o que o cliente já comprou (P1, P2).
    recomendados = {item["product_id"] for item in corpo["recommendations"]}
    assert recomendados.isdisjoint({"P1", "P2"})

    # Cada item traz a atribuição SHAP das quatro entradas do NeuMF.
    for item in corpo["recommendations"]:
        assert set(item["shap"]) == SHAP_FEATURES
        assert all(isinstance(valor, float) for valor in item["shap"].values())


def test_recommend_cold_start(client: TestClient) -> None:
    resposta = client.post("/recommend", json={"customer_name": "Cliente Inexistente"})
    assert resposta.status_code == 200
    corpo = resposta.json()

    assert corpo["cold_start"] is True
    assert len(corpo["recommendations"]) > 0
    # Sem embedding de cliente não há score neural: a explicação vem vazia.
    for item in corpo["recommendations"]:
        assert item["shap"] == {}
    assert corpo["base_value"] is None


def test_recommend_sem_explicacao(client: TestClient) -> None:
    resposta = client.post(
        "/recommend", json={"customer_name": "Ana", "explain": False}
    )
    assert resposta.status_code == 200
    for item in resposta.json()["recommendations"]:
        assert item["shap"] == {}


def test_recommend_entradas_invalidas(client: TestClient) -> None:
    assert client.post("/recommend", json={"customer_name": ""}).status_code == 422
    assert (
        client.post("/recommend", json={"customer_name": "Ana", "top_k": 0}).status_code
        == 422
    )
    assert (
        client.post(
            "/recommend", json={"customer_name": "Ana", "top_k": 999}
        ).status_code
        == 422
    )
