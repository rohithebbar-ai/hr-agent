"""
API Request/Response Models
───────────────────────────
Pydantic models for FastAPI endpoint validation.
"""

from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """Request body for POST/chat"""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question",
        examples=["How many vacation days do i get?"],
    )
    thread_id: str = Field(
        default="default",
        description="Conversation thread ID for memory",
        examples=["discord_user_123"],
    )

class ChatResponse(BaseModel):
    """Response body for POST/chat"""
    answer: str
    thread_id: str
    sources: list[str] = Field(
        default_factory=list,
        description="Policy sources cited in the answer",
    )

class HealthResponse(BaseModel):
    """Response body for GET /health"""
    status: str = "ok"
    pipeline_loaded: bool = False


class ConversationMessage(BaseModel):
    """Single message in conversation history"""
    role: str  # "user" or "assistant"
    content: str


class ConversationHistory(BaseModel):
    """Response body for GET /sessions/{thread_id}/history"""
    thread_id: str
    messages: list[ConversationMessage]
    message_count: int