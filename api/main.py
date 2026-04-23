"""
VanaciRetain FastAPI Backend
────────────────────────────
REST API for the HR Policy Assistant.

Usage:
    uvicorn api.main:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

load_dotenv()

app = FastAPI(
    title="VanaciPrime HR Assistant API",
    description=(
        f"REST API for the VanaciPrime HR Policy Assistant. "
        "Uses a LangGraph agentic RAG pipeline with query "
        "decomposition, document grading, and grounding checks."
    ),
    version="1.0.0",
)

# ── CORS (allow Streamlit and Discord to call the API) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins = [
        "http://localhost:8501",   # Streamlit
        "http://localhost:3000",   # Any local frontend
        "*",                       # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint - redirect to docs"""
    return {
        "service": "VanaciPrime HR assistant",
        "docs": "/docs",
        "health": "/api/v1/health"
    }