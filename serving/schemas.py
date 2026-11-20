"""
Contratos de entrada/saída (Pydantic) do serviço de inferência.

Os modelos aqui definem o schema OpenAPI exposto em ``/docs`` e validam as
requisições antes de qualquer chamada ao modelo.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    """Uma recomendação individual, com a explicação SHAP do score de ranking."""

    ranking: int = Field(description="Posição na lista (1 = melhor).")
    product_id: str = Field(description="Identificador do produto.")
    product_name: str = Field(description="Nome do produto.")
    category: str = Field(description="Categoria do produto.")
    subcategory: str = Field(description="Sub-categoria do produto.")
    score: float = Field(
        description="Score usado no ranking (híbrido popularity-first quando aplicável)."
    )
    shap: dict[str, float] = Field(
        default_factory=dict,
        description="Contribuição de cada entrada (cliente, produto, categoria, subcategoria) "
        "para o score híbrido efetivamente usado no ranking. Vazio em cold start "
        "ou quando ``explain=false``.",
    )


class RecommendRequest(BaseModel):
    """Requisição de recomendação para um cliente."""

    customer_name: str = Field(
        min_length=1,
        description="Nome do cliente (deve constar na base de treino para personalização).",
    )
    top_k: int = Field(
        default=7, ge=1, le=50, description="Quantidade de itens a recomendar."
    )
    explain: bool = Field(
        default=True,
        description="Se verdadeiro, calcula a explicação SHAP por item (custo adicional).",
    )


class RecommendResponse(BaseModel):
    """Lista ranqueada de recomendações para um cliente."""

    customer_name: str
    cold_start: bool = Field(
        description="Verdadeiro quando o cliente é desconhecido (fallback Most-Popular)."
    )
    alpha: float = Field(
        description="Peso do sinal personalizado do NeuMF no score híbrido."
    )
    ranking_mode: Literal["popularity", "hybrid"] = Field(
        description="Estratégia ativa: popularidade pura ou combinação com NeuMF."
    )
    top_k: int
    base_value: float | None = Field(
        default=None,
        description="Valor-base do SHAP (score médio esperado); nulo quando não há explicação.",
    )
    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    """Estado do serviço e dos artefatos carregados."""

    status: str
    model_loaded: bool
    num_products: int = 0
    num_customers: int = 0
    alpha: float = 0.0
