# Multi-stage build for smaller image
FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

WORKDIR /app

# Copy dependency files first (Docker caches this layer)
COPY pyproject.toml uv.lock README.md ./

# Install CPU-only PyTorch
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu --system

# Install Python dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Download the embedding model at build time
# so it's cached in the image (not downloaded on every container start)
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start FastAPI with uvicorn
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]