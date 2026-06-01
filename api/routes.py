"""
API Routes
──────────
All FastAPI endpoint definitions.
The pipeline is injected via FastAPI's dependency system.
"""

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from starlette.requests import Request
from pathlib import Path
from datetime import datetime
from sqlalchemy import desc
from rag.db import SessionLocal, Document, DocumentEvalResult
from agents.pipeline import PolicyAgentPipeline
from api.guardrails.cache import cache_clear, cache_get, cache_set, cache_stats
from api.guardrails.guardrails import(
    redact_pii_logging,
    sanitize_output,
    validate_input,
)
from api.guardrails.limiter import limiter
from api.redis_client import is_redis_available,get_redis
from api.schemas import(
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    ConversationMessage,
    HealthResponse,
)
from rag.db import SessionLocal, Document

load_dotenv()

UPLOAD_DIR = Path("data/hr_documents/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()
logger = logging.getLogger("hragent.api")

# ---- Pipeline singleton ------
@lru_cache(maxsize=1)
def get_pipeline() -> PolicyAgentPipeline:
    """Build the agent pipeline once and reuse"""
    print("[API] Creating pipeline instance")
    return PolicyAgentPipeline()


# ══════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check for load balancers
    returns whether the pipeline is loaded and ready.
    """
    try:
        loaded = get_pipeline.cache_info().currsize > 0
    except Exception:
        loaded = False

    return {
        "status":"ok",
        "pipeline_loaded": loaded,
        "redis_connected": is_redis_available(),
    }

# ------------ cache --------------

@router.get("/cache/stats")
async def get_cache_stats():
    """Cache hit/miss statistics for monitoring."""
    return cache_stats()


@router.delete("/cache")
async def clear_response_cache():
    """Clear all cached responses. Use after re-ingesting documents."""
    cache_clear()
    return {"status": "cleared"}

# ══════════════════════════════════════════════════
# MAIN CHAT ENDPOINT
# ══════════════════════════════════════════════════

@router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest):
    """
    Main chat endpoint with full guardrails.

    Flow:
    1. Validate input (length, injection, special chars)
    2. Check Redis cache for repeat questions
    3. Run LangGraph pipeline
    4. Sanitize output (strip URLs, cap length)
    5. Cache the response in Redis
    6. Log with PII redaction
    """
    # ── Step 1: Input validation ──
    is_valid, error_msg = validate_input(body.question)
    if not is_valid:
        logger.warning(
            f"[GUARDRAIL] Rejected: {error_msg} | "
            f"Input: {redact_pii_logging(body.question[:50])}"
        )
        raise HTTPException(status_code=400, detail=error_msg)

    # ── Step 2: Check cache ──
    cached = cache_get(body.question)
    if cached:
        logger.info(
            f"[CACHE HIT] "
            f"{redact_pii_logging(body.question[:50])}"
        )
        return ChatResponse(
            answer=cached["answer"],
            thread_id=body.thread_id,
            sources=cached["sources"],
        )

    # ── Step 3: Run the agent ──
    pipeline = get_pipeline()

    try:
        graph = pipeline.create_agent()
        config = {
            "configurable": {"thread_id": body.thread_id},
            "metadata": {"source": "api"},
        }

        final_state = graph.invoke(
            {"question": body.question},
            config=config,
        )

        answer = final_state.get("answer", "")

        # Extract sources only for RAG responses
        sources = []
        question_category = final_state.get(
            "question_category", ""
        )

        if question_category == "RAG":
            graded_docs = final_state.get("graded_documents", [])
            seen_titles = set()
            for doc in graded_docs:
                doc_title = doc.metadata.get("document_title", "")
                section = doc.metadata.get("section_title", "")
                if doc_title and doc_title not in seen_titles:
                    seen_titles.add(doc_title)
                    sources.append(f"{doc_title} ({section})" if section else doc_title)

    except Exception as e:
        logger.error(
            f"[AGENT ERROR] {type(e).__name__}: "
            f"{redact_pii_logging(str(e))}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}",
        )

    # ── Step 4: Sanitize output ──
    answer = sanitize_output(answer)

    # ── Step 5: Cache the response ──
    if question_category == "RAG":
        cache_set(body.question, answer, sources)

    # ── Step 6: Log with PII redaction ──
    logger.info(
        f"[CHAT] "
        f"Q: {redact_pii_logging(body.question[:80])} | "
        f"A: {redact_pii_logging(answer[:80])} | "
        f"Category: {question_category} | "
        f"Sources: {len(sources)} | "
        f"Cached: {question_category == 'RAG'}"
    )

    return ChatResponse(
        answer=answer,
        thread_id=body.thread_id,
        sources=sources,
    )


# ══════════════════════════════════════════════════
# SESSION MANAGEMENT
# ══════════════════════════════════════════════════


@router.get(
    "/sessions/{thread_id}/history",
    response_model=ConversationHistory,
)
async def get_session_history(thread_id: str):
    """
    Get conversation history for a thread.
    Reads from the LangGraph MemorySaver checkpointer.
    """
    pipeline = get_pipeline()

    try:
        graph = pipeline.create_agent()
        config = {"configurable": {"thread_id": thread_id}}

        # Get the current state snapshot
        state = graph.get_state(config)

        if state.values is None:
            return ConversationHistory(
                thread_id=thread_id,
                messages=[],
                message_count=0,
            )

        # Extract messages from state
        raw_messages = state.values.get("messages", [])
        messages = []

        for msg in raw_messages:
            if hasattr(msg, "content") and msg.content:
                role = (
                    "user" if msg.type == "human"
                    else "assistant"
                )
                messages.append(ConversationMessage(
                    role=role,
                    content=msg.content,
                ))

        return ConversationHistory(
            thread_id=thread_id,
            messages=messages,
            message_count=len(messages),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}",
        )

@router.delete("/sessions/{thread_id}")
async def clear_session(thread_id: str):
    """
    Clear conversation history for a thread.
    Note: MemorySaver is in-memory, so this clears on next restart.
    For now, we create a fresh thread by returning a success message.
    """
    # MemorySaver doesn't expose a delete method directly.
    # The practical approach: tell the client to use a new thread_id.
    return {
        "status": "cleared",
        "thread_id": thread_id,
        "message": "Session cleared. Use a new thread_id for a fresh conversation.",
    }


# ══════════════════════════════════════════════════
# Admin behaviour - Discord connection
# ══════════════════════════════════════════════════


@router.get("/admin/behavior")
async def get_behavior_prompt():
    default = "You are a helpful HR assistant"
    if is_redis_available():
        r = get_redis()
        prompt = r.get("hrassistant:behavior_prompt")
        return {"prompt": prompt or default}
    return {"prompt": default}

@router.post("/admin/behavior")
async def set_behavior_prompt(prompt:str = Form(...)):
    if is_redis_available():
        r = get_redis()
        r.set("hrassistant:behavior_prompt", prompt)
    return {"status": "saved", "prompt_length": len(prompt)}


@router.post("/admin/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    document_type: str = Form(default="Employee Handbook"),
    tenant_id: str = Form(default="vanaciprime"),
    sample_questions: str = Form(default=""),
    use_llamaparse: bool = Form(default=True), # Admin Toggle
):
    """
    Upload document for async ingestion.

    use_llamaparse: True  = try LlamaParse first (best quality, costs credits)
                   False = use Unstructured + PyMuPDF (free, good for text docs)

    The worker will automatically fall back to Unstructured if LlamaParse
    fails or quota is exceeded, regardless of this setting.
    """
    import hashlib
    import uuid
    import json
    job_id = str(uuid.uuid4())[:8]
    safe_filename = f"{job_id}_{file.filename}"
    file_path = UPLOAD_DIR / safe_filename

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    
    document_id = hashlib.sha256(content).hexdigest()[:16]

    # Create neon record
    db = SessionLocal()
    try:
        existing = db.query(Document).filter(
            Document.document_id == document_id
        ).first()

        if existing:
            existing.filename = file.filename
            existing.status = "pending"
            existing.upload_date = datetime.utcnow()
            if sample_questions:
                existing.sample_questions = sample_questions
        else:
            doc = Document(
                document_id = document_id, 
                filename = file.filename,
                file_type = Path(file.filename).suffix.lstrip("."),
                document_type = document_type,
                tenant_id = tenant_id,
                status = "pending",
                sample_questions = sample_questions,
            )
            db.add(doc)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[UPLOAD] DB insert failed: {e}")
    finally:
        db.close()

    # Job queue
    job = {
        "job_id": job_id,
        "file_path": str(file_path),
        "filename": file.filename,
        "document_id": document_id,
        "tenant_id": tenant_id,
        "document_type": document_type,
        "size_bytes": len(content),
        "use_llamaparse": use_llamaparse,
    }

    from api.redis_client import get_redis, is_redis_available
    if is_redis_available():
        r = get_redis()
        r.rpush("ingestion_jobs", json.dumps(job))
        r.setex(f"job:{job_id}", 86400, json.dumps({"status": "queued"}))

    return {
        "job_id": job_id,
        "document_id": document_id,
        "filename": file.filename,
        "status": "queued",
        "poll_url": f"/api/v1/admin/status/{job_id}",
    }


@router.get("/admin/llamaparse/usage")
async def get_llamaparse_usage(request: Request):
    """Check remaining llamaparse credits"""
    try:
        import httpx
        api_key = os.environ.get("LLAMAPARSE_API_KEY", "")
        if not api_key:
            return {"error": "LLAMAPARSE_API_KEY not configured"}

        # LLamaparse usage endpoint
        r = httpx.get(
            "https://api.cloud.llamaindex.ai/api/v1/parsing/usage",
            headers = {"Authorization": f"Bearer {api_key}"},
            timeout = 10.0
        )
        if r.status_code == 200:
            return r.json()
        return {"error": f"API returned {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/admin/documents")
async def list_documents(
    request: Request,
    tenant_id: str = "vanaciprime",
):
    """List all ingested documents with status and latest quality score."""
    db = SessionLocal()
    try:
        documents = (
            db.query(Document)
            .filter(Document.tenant_id == tenant_id)
            .order_by(desc(Document.upload_date))
            .all()
        )

        response_docs = []
        for doc in documents:
            latest_eval = (
                db.query(DocumentEvalResult)
                .filter(DocumentEvalResult.document_id == doc.document_id)
                .order_by(desc(DocumentEvalResult.eval_date))
                .first()
            )

            eval_data = None
            if latest_eval:
                eval_data = {
                    "pass_rate": latest_eval.pass_rate,
                    "questions_tested": latest_eval.questions_tested,
                    "questions_passed": latest_eval.questions_passed,
                    "status": latest_eval.status,
                    "eval_date": str(latest_eval.eval_date) if latest_eval.eval_date else None,
                }

            response_docs.append({
                "document_id": doc.document_id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "document_type": doc.document_type,
                "chunk_count": doc.chunk_count,
                "upload_date": str(doc.upload_date) if doc.upload_date else None,
                "status": doc.status,
                "error_message": doc.error_message,
                "loader_used": doc.loader_used,
                "eval": eval_data,
            })

        return {"documents": response_docs, "total": len(response_docs)}

    finally:
        db.close()


@router.delete("/admin/documents/{document_id}")
async def delete_document(request: Request, document_id: str):
    """
    Delete document — removes chunks from Qdrant Cloud AND Neon record.
    Eval results are cascade deleted via the SQLAlchemy relationship.
    """
    # Remove from Qdrant Cloud
    try:
        from rag.pipeline.upserter import get_client, COLLECTION_NAME
        from qdrant_client.models import (
            FilterSelector, Filter, FieldCondition, MatchValue
        )
        client = get_client()
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(
                filter=Filter(must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ])
            ),
        )
        print(f"[DELETE] Qdrant chunks removed for: {document_id}")
    except Exception as e:
        print(f"[DELETE] Qdrant delete failed: {e}")

    # Remove from Neon
    db = SessionLocal()
    filename = "unknown"
    try:
        doc = db.query(Document).filter(
            Document.document_id == document_id
        ).first()

        if doc:
            filename = doc.filename
            db.delete(doc)    # Cascades to eval_results
            db.commit()
            print(f"[DELETE] Neon record removed: {filename}")
        else:
            print(f"[DELETE] Document not found in Neon: {document_id}")
    except Exception as e:
        db.rollback()
        print(f"[DELETE] Neon delete failed: {e}")
    finally:
        db.close()

    return {"status": "deleted", "document_id": document_id, "filename": filename}


@router.get("/admin/status/{job_id}")
async def get_job_status(request: Request, job_id: str):
    """Poll ingestion job status from Redis."""
    if is_redis_available():
        r = get_redis()
        data = r.get(f"job:{job_id}")
        if data:
            import json as _json
            parsed = _json.loads(data)
            return {
                "status": parsed.get("status", "unknown"),
                "job_id": job_id,
                "result": parsed.get("result"),
            }
    return {"status": "unknown", "job_id": job_id}
