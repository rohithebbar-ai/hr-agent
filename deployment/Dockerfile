FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (dependency manager)
RUN pip install uv

WORKDIR /app

# Copy dependency files first (stable layer for caching)
COPY pyproject.toml uv.lock README.md ./

# Install Python dependencies into .venv
RUN uv sync --frozen --no-dev

# Ensure the virtual environment is used at runtime
ENV PATH="/app/.venv/bin:$PATH"

# Download embedding model once during build (cached layer)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code (changes frequently)
COPY . .

# Expose API port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run application WITHOUT uv run (prevents reinstall on startup)
CMD ["sh", "-c", "python scripts/aws_secrets.py && uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2"]