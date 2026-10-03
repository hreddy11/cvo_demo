# syntax=docker/dockerfile:1

# ---------- builder: resolve + install runtime deps with uv ----------
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# 1) dependencies only (cached unless pyproject.toml / project metadata change)
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project

# 2) our package, installed non-editable so the runtime image does not need the source tree
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-editable

# ---------- runtime: slim, non-root, no build tools ----------
FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/models/model.joblib \
    FEATURES_PATH=/data/serving_features.parquet \
    META_PATH=/data/feature_meta.json

USER appuser
EXPOSE 8000

# python (not curl) because slim images ship without curl; /ready fails until model + features are loaded
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "cvo_demo.service:app", "--host", "0.0.0.0", "--port", "8000"]
