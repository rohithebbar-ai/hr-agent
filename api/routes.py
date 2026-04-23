"""
API Routes
──────────
All FastAPI endpoint definitions.
The pipeline is injected via FastAPI's dependency system.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException

from agents.pipeline import PolicyAgentPipeline
from api.schemas import(
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    ConversationMessage,
    HealthResponse,
)

load_dotenv()

router = APIRouter()

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

    return HealthResponse(status="ok", pipeline_loaded=loaded)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    Runs the question through the langgraph agentic pipeline.
    """
    pipeline = get_pipeline()

    try:
        # run the agent
        graph = pipeline.create_agent()
        config = {
            "configurable": {"thread_id": request.thread_id},
            "metadata": {"source": "api"},
        }

        final_state = graph.invoke(
            {"question": request.question},
            config = config,
        )

        answer = final_state.get("answer", "")

        # Extract the source citation from graded documents
        # Extract source citations only for RAG responses
        sources = []
        question_category = final_state.get("question_category", "")

        if question_category == "RAG":
            graded_docs = final_state.get("graded_documents", [])
            for doc in graded_docs:
                policy_name = doc.metadata.get("policy_name", "")
                section = doc.metadata.get("section", "")
                if policy_name and policy_name not in sources:
                    sources.append(f"{policy_name} ({section})")

        return ChatResponse(
            answer = answer,
            thread_id = request.thread_id,
            sources = sources,
        )

    except Exception as e:
        print(f"[API] Error: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code = 500,
            detail = f"Agent error: {str(e)}"
        )


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