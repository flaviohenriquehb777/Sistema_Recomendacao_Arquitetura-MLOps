"""
Aplicação FastAPI do serviço de inferência.

Expõe dois endpoints:

* ``GET  /health``    — estado do serviço e dos artefatos carregados;
* ``POST /recommend`` — top-k personalizado com explicação SHAP por item.

Execução local::

    uv run uvicorn serving.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request

from serving.schemas import HealthResponse, RecommendRequest, RecommendResponse
from serving.service import RecommenderService

logger = logging.getLogger("serving")


def get_service(request: Request) -> RecommenderService:
    """Dependência que devolve o serviço carregado (ou 503 se indisponível).

    Definida no módulo (e não dentro de ``create_app``) para que a anotação
    ``Annotated[..., Depends(get_service)]`` seja resolvível por ``get_type_hints``.
    """
    svc = getattr(request.app.state, "service", None)
    if svc is None:
        raise HTTPException(
            status_code=503, detail="Modelo indisponível: artefatos não carregados."
        )
    return svc


def create_app(service: RecommenderService | None = None) -> FastAPI:
    """Cria a aplicação FastAPI.

    Se ``service`` for informado (por exemplo, nos testes), ele é usado
    diretamente; caso contrário os artefatos são carregados no *startup*.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is not None:
            app.state.service = service
        else:
            try:
                app.state.service = RecommenderService.load()
                logger.info("Artefatos do recomendador carregados com sucesso.")
            except Exception:
                logger.exception(
                    "Falha ao carregar os artefatos; /recommend responderá 503."
                )
                app.state.service = None
        yield

    app = FastAPI(
        title="Sistema de Recomendação — Serviço de Inferência",
        description="API de referência para o recomendador híbrido popularity-first (NeuMF).",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["infra"])
    def health() -> HealthResponse:
        svc = getattr(app.state, "service", None)
        if svc is None:
            return HealthResponse(status="unavailable", model_loaded=False)
        return HealthResponse(
            status="ok",
            model_loaded=True,
            num_products=svc.num_products,
            num_customers=svc.num_customers,
            alpha=svc.alpha,
        )

    @app.post("/recommend", response_model=RecommendResponse, tags=["recomendação"])
    def recommend(
        payload: RecommendRequest,
        svc: Annotated[RecommenderService, Depends(get_service)],
    ) -> RecommendResponse:
        resultado = svc.recommend(
            payload.customer_name, top_k=payload.top_k, explain=payload.explain
        )
        return RecommendResponse(**resultado)

    return app


app = create_app()
