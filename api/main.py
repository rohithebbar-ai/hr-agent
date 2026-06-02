"""
HR-copilot FastAPI Backend
────────────────────────────
REST API for the HR Policy Assistant.

Usage:
    uvicorn api.main:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""
import os

# Load AWS Secrets Manager into os.environ BEFORE any other imports
# This must run in the same process as uvicorn
_secret_name = os.environ.get("AWS_SECRET_NAME", "hragent/api-keys")
_aws_region = os.environ.get("AWS_REGION", "ap-south-1")

if _secret_name and os.environ.get("ENVIRONMENT") == "prod":
    try:
        import json
        import boto3
        client = boto3.client("secretsmanager", region_name=_aws_region)
        response = client.get_secret_value(SecretId=_secret_name)
        secrets = json.loads(response["SecretString"])
        for key, value in secrets.items():
            if value and key not in os.environ:
                os.environ[key] = value
        print(f"[SECRETS] Loaded {len(secrets)} keys from Secrets Manager")
    except Exception as e:
        print(f"[SECRETS] Failed: {e}")

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.routes import router

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Warming up retrieval models...")
    try:
        from rag.retriever_enterprise import get_dense_model, get_sparse_model, get_ranker
        get_dense_model()
        get_sparse_model()
        get_ranker()
        print("[STARTUP] Models ready — first request will be fast")
    except Exception as e:
        print(f"[STARTUP] Warmup failed (non-fatal): {e}")
    yield
    print("[SHUTDOWN] FastAPI shutting down")


app = FastAPI(
    title="HR Assistant API",
    description=(
        "REST API for the HR Policy Assistant. "
        "Uses a LangGraph agentic RAG pipeline with query "
        "decomposition, document grading, and grounding checks."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit
        "http://localhost:3000",  # Any local frontend
        "*",                      # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router, prefix="/api/v1")


@app.get("/admin", include_in_schema=False)
async def admin_panel():
    """Serve the admin HTML panel."""
    return FileResponse("admin.html")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "HR assistant",
        "docs": "/docs",
        "health": "/api/v1/health",
    }