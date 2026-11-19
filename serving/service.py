"""
Serviço de inferência do sistema de recomendação.

Carrega, uma única vez, os artefatos produzidos pelo estágio DVC
``treinamento_modelo`` e oferece as operações reutilizadas pela API:

* ``recommend`` — top-k personalizado (híbrido popularity-first), excluindo os
  itens já vistos pelo cliente e caindo no *baseline* Most-Popular para clientes
  desconhecidos (cold start);
* explicação **SHAP** por item, atribuindo o score neural às quatro entradas do
  NeuMF (cliente, produto, categoria, subcategoria) — a mesma abordagem de
  ``notebooks/05_SHAP_Avancado.ipynb`` (``shap.KernelExplainer``).

Este módulo NÃO importa ``src.config.paths`` (que exige o CSV bruto). Os
caminhos dos artefatos vêm de ``MODELS_DIR`` / ``DATA_PATH`` (variáveis de
ambiente), o que mantém a imagem de container enxuta.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf

from src.config.model_utils import (
    TOP_K,
    build_item_catalog,
    recommend_products,
)

RANDOM_SEED = 42

# Nomes das quatro entradas do NeuMF, na mesma ordem usada no treino.
SHAP_FEATURES: tuple[str, ...] = ("cliente", "produto", "categoria", "subcategoria")
DEFAULT_TOP_K = TOP_K
DEFAULT_BACKGROUND_SIZE = 50
DEFAULT_SHAP_NSAMPLES = 100

MODEL_FILENAME = "best_model_recomendacao_otimizado.keras"
HYBRID_FILENAME = "hybrid_config_otimizado.pkl"
ENCODER_FILENAMES = {
    "customer": "customer_encoder_otimizado.pkl",
    "product": "product_encoder_otimizado.pkl",
    "category": "category_encoder_otimizado.pkl",
    "subcategory": "subcategory_encoder_otimizado.pkl",
}


class RecommenderService:
    """Encapsula o modelo treinado, os encoders, o catálogo e a explicação SHAP."""

    def __init__(
        self,
        model: tf.keras.Model,
        df: pd.DataFrame,
        encoders: dict[str, object],
        catalog: tuple[np.ndarray, np.ndarray, np.ndarray],
        alpha: float,
        popularity: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        background_size: int = DEFAULT_BACKGROUND_SIZE,
        shap_nsamples: int = DEFAULT_SHAP_NSAMPLES,
    ) -> None:
        self.model = model
        self.df = df
        self.encoders = encoders
        self.catalog = catalog
        self.alpha = float(alpha)
        self.popularity = popularity
        self.top_k = int(top_k)
        self.shap_nsamples = int(shap_nsamples)

        self.product_ids, self.category_ids, self.subcategory_ids = catalog
        self.num_products = len(self.product_ids)
        self.num_customers = len(encoders["customer"].classes_)

        self._background = self._build_background(background_size)
        self._explainer: shap.KernelExplainer | None = None

    # ------------------------------------------------------------------ #
    # Construção
    # ------------------------------------------------------------------ #
    @classmethod
    def load(
        cls, models_dir: str | Path | None = None, data_path: str | Path | None = None
    ) -> RecommenderService:
        """Carrega os artefatos do disco (modelo, encoders, config do híbrido e dados)."""
        models_dir = Path(models_dir or os.environ.get("MODELS_DIR", "models"))
        data_path = Path(
            data_path or os.environ.get("DATA_PATH", "dados/dados_tratados.parquet")
        )

        model = tf.keras.models.load_model(models_dir / MODEL_FILENAME)
        config = joblib.load(models_dir / HYBRID_FILENAME)
        encoders = {
            nome: joblib.load(models_dir / arquivo)
            for nome, arquivo in ENCODER_FILENAMES.items()
        }

        df = pd.read_parquet(data_path)
        catalog = build_item_catalog(
            df, encoders["product"], encoders["category"], encoders["subcategory"]
        )
        return cls(
            model=model,
            df=df,
            encoders=encoders,
            catalog=catalog,
            alpha=config["alpha"],
            popularity=config["popularity"],
        )

    def _build_background(self, size: int) -> np.ndarray:
        """Amostra determinística de interações observadas, usada como *background* do SHAP."""
        n = min(int(size), len(self.df))
        amostra = self.df.sample(n=n, random_state=RANDOM_SEED)
        return np.column_stack(
            [
                self.encoders["customer"].transform(amostra["Customer Name"]),
                self.encoders["product"].transform(amostra["Product ID"]),
                self.encoders["category"].transform(amostra["Category"]),
                self.encoders["subcategory"].transform(amostra["Sub-Category"]),
            ]
        ).astype("int64")

    # ------------------------------------------------------------------ #
    # Inferência
    # ------------------------------------------------------------------ #
    def recommend(
        self, customer_name: str, top_k: int | None = None, explain: bool = True
    ) -> dict:
        """Gera o top-k do cliente e (opcionalmente) a explicação SHAP por item."""
        top_k = int(top_k or self.top_k)
        customer_id = self._encode_customer(customer_name)
        cold_start = customer_id is None

        tabela = recommend_products(
            self.model,
            customer_name,
            self.df,
            self.encoders["customer"],
            self.encoders["product"],
            self.catalog,
            top_k=top_k,
            popularity=self.popularity,
            alpha=self.alpha,
        )

        atribuicoes: list[dict[str, float]] = [{} for _ in range(len(tabela))]
        base_value: float | None = None
        if explain and not cold_start and len(tabela):
            atribuicoes, base_value = self._explain(customer_id, tabela)

        recomendacoes = [
            {
                "ranking": int(linha["ranking"]),
                "product_id": str(linha["Product ID"]),
                "product_name": str(linha["Product Name"]),
                "category": str(linha["Category"]),
                "subcategory": str(linha["Sub-Category"]),
                "score": float(linha["score"]),
                "shap": atribuicoes[i],
            }
            for i, (_, linha) in enumerate(tabela.iterrows())
        ]

        return {
            "customer_name": customer_name,
            "cold_start": cold_start,
            "alpha": self.alpha,
            "top_k": top_k,
            "base_value": base_value,
            "recommendations": recomendacoes,
        }

    def _encode_customer(self, customer_name: str) -> int | None:
        """Codifica o cliente; devolve ``None`` quando ele é desconhecido (cold start)."""
        try:
            return int(self.encoders["customer"].transform([customer_name])[0])
        except ValueError:
            return None

    @property
    def explainer(self) -> shap.KernelExplainer:
        """``KernelExplainer`` construído de forma preguiçosa sobre o score neural."""
        if self._explainer is None:
            self._explainer = shap.KernelExplainer(self._score, self._background)
        return self._explainer

    def _score(self, X: np.ndarray) -> np.ndarray:
        """Função de score neural (probabilidade) das quatro entradas do NeuMF."""
        X = np.asarray(X)
        return self.model.predict(
            [
                X[:, 0].reshape(-1, 1),
                X[:, 1].reshape(-1, 1),
                X[:, 2].reshape(-1, 1),
                X[:, 3].reshape(-1, 1),
            ],
            verbose=0,
        ).ravel()

    def _explain(
        self, customer_id: int, tabela: pd.DataFrame
    ) -> tuple[list[dict[str, float]], float]:
        """Atribui o score neural de cada item recomendado às quatro entradas (SHAP)."""
        entradas = np.column_stack(
            [
                np.full(len(tabela), customer_id, dtype="int64"),
                self.encoders["product"].transform(tabela["Product ID"]),
                self.encoders["category"].transform(tabela["Category"]),
                self.encoders["subcategory"].transform(tabela["Sub-Category"]),
            ]
        ).astype("int64")

        valores = np.asarray(
            self.explainer.shap_values(entradas, nsamples=self.shap_nsamples)
        )
        if valores.ndim == 3:  # (n, features, saída) em algumas versões do SHAP
            valores = valores[..., 0]

        atribuicoes = [
            {
                nome: round(float(valores[i, j]), 6)
                for j, nome in enumerate(SHAP_FEATURES)
            }
            for i in range(len(tabela))
        ]
        base_value = round(float(np.ravel(self.explainer.expected_value)[0]), 6)
        return atribuicoes, base_value
