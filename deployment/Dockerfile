FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-dev

COPY . .

ENV FASTD_CACHE_PATH=/app/.fastembed_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/.sentence_transformers

RUN uv run python -c "
from sentence_transformers import SentenceTransformer
print('Downloading all-MiniLM-L6-v2...')
SentenceTransformer('all-MiniLM-L6-v2')
print('Done')
"

RUN uv run python -c "
from fastembed import SparseTextEmbedding
print('Downloading SPLADE...')
model = SparseTextEmbedding(model_name='prithivida/Splade_PP_en_v1')
list(model.embed(['warmup']))
print('Done')
"

RUN uv run python -c "
from flashrank import Ranker
print('Downloading flashrank reranker...')
Ranker(model_name='ms-marco-MiniLM-L-12-v2')
print('Done')
"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["sh", "-c", "uv run python scripts/aws_secrets.py && uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1"]