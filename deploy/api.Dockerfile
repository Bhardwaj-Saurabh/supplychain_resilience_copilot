# API + in-process agent runtime (demo profile boots with no external services).
# Build context is the repo root: docker build -f deploy/api.Dockerfile .
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    SCRC_PROFILE=demo

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY packages ./packages

# Demo profile needs the API, orchestration, ML, and data layers.
RUN uv pip install --system -e ".[api,orchestration,ml,data]"

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "scrc.app.composition:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
