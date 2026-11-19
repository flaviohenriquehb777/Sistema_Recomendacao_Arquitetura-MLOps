# syntax=docker/dockerfile:1
# =============================================================================
# Serviço de inferência — Sistema de Recomendação (NeuMF híbrido popularity-first)
# -----------------------------------------------------------------------------
# Build em dois estágios:
#   1) builder : resolve o ambiente com uv a partir do lock travado (--frozen);
#   2) runtime : imagem enxuta (sem uv, sem cache) rodando como usuário não-root.
#
# Atenção: os artefatos de modelo e os dados tratados são versionados via DVC
# (não no git). Rode `dvc pull` antes do build para que `models/` e
# `dados/dados_tratados.parquet` existam no contexto de build.
#
#   docker build -t sistema-recomendacao:latest .
#   docker run --rm -p 8000:8000 sistema-recomendacao:latest
# =============================================================================

# --- Estágio 1: builder ------------------------------------------------------
FROM python:3.11-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# Binário do uv, fixado para reprodutibilidade.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

WORKDIR /app

# 1) Dependências primeiro (camada cacheável: só muda quando o lock muda).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

# 2) Código-fonte consumido em tempo de execução.
COPY src/ ./src/
COPY serving/ ./serving/

# --- Estágio 2: runtime ------------------------------------------------------
FROM python:3.11-slim AS runtime

# libgomp1: exigida pelos wheels manylinux do scikit-learn (OpenMP).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODELS_DIR=/app/models \
    DATA_PATH=/app/dados/dados_tratados.parquet

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
COPY serving/ ./serving/
COPY models/ ./models/
COPY dados/ ./dados/

# Usuário não-root.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
