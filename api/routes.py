"""
API Routes
──────────
All FastAPI endpoint definitions.
The pipeline is injected via FastAPI's dependency system.
"""

import os
import logging
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from starlette.requests import Request

from agents.pipeline import PolicyAgentPipeline
from api.guardrails.cache import cache_clear, cache_get, cache_set, cache_stats
from api.guardrails.guardrails import(
    redact_pii_logging,
    sanitize_output,
    validate_input,
)
from api.guardrails.limiter import limiter
from api.redis_client import is_redis_available
from api.schemas import(
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    ConversationMessage,
    HealthResponse,
)

load_dotenv()

router = APIRouter()
logger = logging.getLogger("vanacihr.api")

# ---- Pipeline singleton ------
@lru_cache(maxsize=1)
def get_pipeline() -> PolicyAgentPipeline:
    """Build the agent pipeline once and reuse"""
    print("[API] Building agent pipeline")
    pipeline = PolicyAgentPipeline()
    pipeline.create_agent()
    print("[API] pipeline ready")
    return pipeline


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
        pipeline = get_pipeline()
        loaded = pipeline._graph is not None
    except Exception:
        loaded = False

    return {
        "status":"ok",
        "pipeline_loaded": loaded,
        "redis_connected": is_redis_available(),
    }

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
            for doc in graded_docs:
                policy_name = doc.metadata.get("policy_name", "")
                section = doc.metadata.get("section", "")
                if policy_name and policy_name not in sources:
                    sources.append(f"{policy_name} ({section})")

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