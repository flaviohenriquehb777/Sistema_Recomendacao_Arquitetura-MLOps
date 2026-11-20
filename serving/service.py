"""
Serviço de inferência do sistema de recomendação.

Carrega, uma única vez, os artefatos produzidos pelo estágio DVC
``treinamento_modelo`` e oferece as operações reutilizadas pela API:

* ``recommend`` — top-k excluindo os itens já vistos, usando o modo escolhido
  pela validação (híbrido ou popularidade) e recorrendo a Most-Popular para
  clientes desconhecidos (cold start);
* explicação **SHAP** por item, atribuindo o score híbrido efetivamente usado
  no ranking às quatro entradas do NeuMF (cliente, produto, categoria,
  subcategoria).

Este módulo NÃO importa ``src.config.paths`` (que exige o CSV bruto). Os
caminhos dos artefatos vêm de ``MODELS_DIR`` / ``DATA_PATH`` (variáveis de
ambiente), o que mantém a imagem de container enxuta.

O serviço é **agnóstico de base**: não assume nenhum conjunto de artefatos fixo.
Quem define o que é servido é ``DATA_PATH``, e o papel do conjunto é declarado
em ``MODEL_VARIANT``. O conjunto disponível hoje — o de ``sc_base_01`` — está
rotulado como ``baseline`` (ver ``docs/ROADMAP_RECOMENDACAO_ESCALA.md``), e não
como modelo principal da API.
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

# Rótulo do papel do conjunto servido (``MODEL_VARIANT``).
DEFAULT_VARIANT = "baseline"
# Usado quando o caminho dos dados não revela a base de origem.
UNKNOWN_BASE_CODE = "desconhecida"

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
        base_code: str = UNKNOWN_BASE_CODE,
        variant: str = DEFAULT_VARIANT,
    ) -> None:
        self.model = model
        self.df = df
        self.encoders = encoders
        self.catalog = catalog
        self.alpha = float(alpha)
        self.popularity = popularity
        self.top_k = int(top_k)
        self.shap_nsamples = int(shap_nsamples)
        self.base_code = str(base_code)
        self.variant = str(variant)

        self.product_ids, self.category_ids, self.subcategory_ids = catalog
        self.num_products = len(self.product_ids)
        self.num_customers = len(encoders["customer"].classes_)
        popularidade_log = np.log1p(np.asarray(popularity, dtype="float64"))
        desvio_popularidade = float(popularidade_log.std())
        self.popularity_z = (popularidade_log - popularidade_log.mean()) / (
            desvio_popularidade if desvio_popularidade else 1.0
        )

        self._background = self._build_background(background_size)

    # ------------------------------------------------------------------ #
    # Construção
    # ------------------------------------------------------------------ #
    @classmethod
    def load(
        cls, models_dir: str | Path | None = None, data_path: str | Path | None = None
    ) -> RecommenderService:
        """Carrega os artefatos do disco (modelo, encoders, config do híbrido e dados).

        A base servida não é fixada em código: vem de ``data_path`` ou da variável
        de ambiente ``DATA_PATH`` (obrigatória), e o código da base é derivado da
        pasta do arquivo (``sc_base_01``, ``sc_base_02``, ...). O papel do conjunto
        é declarado em ``MODEL_VARIANT`` (padrão ``baseline``).
        """
        models_dir = Path(models_dir or os.environ.get("MODELS_DIR", "models"))
        caminho_dados = data_path or os.environ.get("DATA_PATH")
        if not caminho_dados:
            raise ValueError(
                "Base servida não definida: informe DATA_PATH (variável de "
                "ambiente) ou o argumento data_path. O serviço não assume um "
                "conjunto de artefatos fixo."
            )
        data_path = Path(caminho_dados)
        base_code = data_path.parent.name or UNKNOWN_BASE_CODE
        variant = os.environ.get("MODEL_VARIANT", DEFAULT_VARIANT)

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
            base_code=base_code,
            variant=variant,
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
            "ranking_mode": (
                "popularity" if cold_start or self.alpha == 0.0 else "hybrid"
            ),
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

    def _predict_neural(self, X: np.ndarray) -> np.ndarray:
        """Prediz o score neural para as quatro entradas do NeuMF."""
        X = np.asarray(X)
        previsoes = self.model(
            [
                X[:, 0].reshape(-1, 1),
                X[:, 1].reshape(-1, 1),
                X[:, 2].reshape(-1, 1),
                X[:, 3].reshape(-1, 1),
            ],
            training=False,
        )
        return np.asarray(previsoes).ravel()

    def _hybrid_score(
        self, X: np.ndarray, model_mean: float, model_std: float
    ) -> np.ndarray:
        """Reproduz o score de ranking para uma distribuição fixa do cliente."""
        X = np.asarray(X, dtype="int64")
        desvio_modelo = model_std if model_std else 1.0
        score_neural = self._predict_neural(X)
        return self.popularity_z[X[:, 1]] + self.alpha * (
            (score_neural - model_mean) / desvio_modelo
        )

    def _explain(
        self, customer_id: int, tabela: pd.DataFrame
    ) -> tuple[list[dict[str, float]], float]:
        """Atribui o score híbrido de cada item recomendado via SHAP."""
        entradas = np.column_stack(
            [
                np.full(len(tabela), customer_id, dtype="int64"),
                self.encoders["product"].transform(tabela["Product ID"]),
                self.encoders["category"].transform(tabela["Category"]),
                self.encoders["subcategory"].transform(tabela["Sub-Category"]),
            ]
        ).astype("int64")

        entradas_catalogo = np.column_stack(
            [
                np.full(self.num_products, customer_id, dtype="int64"),
                self.product_ids,
                self.category_ids,
                self.subcategory_ids,
            ]
        )
        scores_catalogo = self._predict_neural(entradas_catalogo)
        media_modelo = float(scores_catalogo.mean())
        desvio_modelo = float(scores_catalogo.std())

        def funcao_score(X: np.ndarray) -> np.ndarray:
            return self._hybrid_score(X, media_modelo, desvio_modelo)

        explainer = shap.KernelExplainer(funcao_score, self._background)

        valores = np.asarray(
            explainer.shap_values(entradas, nsamples=self.shap_nsamples)
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
        base_value = round(float(np.ravel(explainer.expected_value)[0]), 6)
        return atribuicoes, base_value
