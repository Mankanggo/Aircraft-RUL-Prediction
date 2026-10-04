# syntax=docker/dockerfile:1
#
# Aircraft engine RUL API: FastAPI + the frozen classical inference pipeline + models/final.
#
#   docker build -t aircraft-rul-api:latest .                       # runtime image (default, last stage)
#   docker build --target test -t aircraft-rul-api:test .          # same layers + test tools, runs the API tests
#   docker run --rm -p 8000:8000 --read-only --tmpfs /tmp aircraft-rul-api:latest
#
# The pickled preprocessors reference the module path src.components.data_transformation, so the
# src/ package is copied unchanged and PYTHONPATH=/app.

# Python 3.13.7 = the interpreter the frozen artifacts were trained with (models/final/manifest.json).
ARG PYTHON_IMAGE=python:3.13.7-slim

# ---------------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    RUL_MODEL_DIR=/app/models/final \
    RUL_CONFIG_DIR=/app/configs

# libgomp1: OpenMP runtime required by LightGBM (and used by XGBoost)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Exact pins: ML libraries = manifest versions (the API refuses to start otherwise).
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Runtime files only (see .dockerignore); frozen artifacts copied unchanged.
COPY src/ src/
COPY configs/ configs/
COPY models/final/ models/final/

# Build-time verification: fail the build unless all four frozen predictors load via
# RULPredictor.load(verify=True) - artifact SHA-256 hashes vs manifest.json - and the installed
# library versions and frozen configuration match. Nothing is trained or regenerated.
RUN python -c "\
from src.api.settings import Settings; \
from src.api.model_registry import load_registry; \
s = Settings.from_env(); r = load_registry(s.model_dir, s.config_dir); \
print('frozen bundle verified:', r.config_version, r.checks, {k: (m.family, m.rul_cap) for k, m in r.models.items()})"

# ---------------------------------------------------------------------------------------------
# Test target: identical runtime layers plus test tools and the API tests / golden fixtures.
# Never used for the runtime image.
FROM base AS test
RUN pip install --no-cache-dir pytest==9.1.1 httpx==0.28.1
COPY tests/conftest.py tests/conftest.py
COPY tests/api/ tests/api/
COPY tests/fixtures/ tests/fixtures/
CMD ["python", "-m", "pytest", "tests/api", "-q", "-p", "no:cacheprovider", "-W", "ignore::DeprecationWarning"]

# ---------------------------------------------------------------------------------------------
# Runtime target (default): non-root, no file writes, healthy only when /ready returns 200.
FROM base AS runtime

RUN groupadd --system --gid 10001 rul \
    && useradd --system --uid 10001 --gid rul --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin rul
USER 10001:10001

EXPOSE 8000

# /ready = all four predictors loaded and hash/version/config checks passed (python: no curl in slim)
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
