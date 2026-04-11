"""
Schemas for the Policy Agent.
 
- PolicyAgentState: the state that flows between nodes
- Structured output schemas: used for reliable LLM parsing
"""

from typing import Annotated, List, Literal, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict



 
# ───────────────────────────────────────────
# STRUCTURED OUTPUT SCHEMAS
# ───────────────────────────────────────────
# These are used with llm.with_structured_output(Schema)
# to force the LLM to return exactly the fields we want.
# No more parsing "yes"/"no" strings or broken JSON.

class RouteQuery(BaseModel):
    """Routes a user question to CHAT or RAG pipeline."""
    category: Literal["chat", "RAG"] = Field(
        description=(
            "CHAT for greetings, small talk, or general questions"
            "not requiring document retrieval"
            "RAG for HR policy, benefits, leave, workplace conduct,"
            "or anything requiring the HR handbook"
        )
    )

class QueryDecomposition(BaseModel):
    """Decomposes a user query into sub-queries for multi-hop retrieval."""
    sub_queries: List[str] = Field(
        description=(
            "List of sub-queries. If the original query is simple, "
            "return it as a single-item list. If its a comparision "
            "or multi-hop question, decompose it into 2-4 focused sub-queries"
        ),
        min_length=1,
        max_length=4
    )

class DocumentGrade(BaseModel):
    """Grades whether a document is relevant to a query."""
    is_relevant: Literal["yes", "no"] = Field(
        description=(
            "'yes' if the document directly helps answer the query,"
            "'no' otherwise"
        )
    )

class GroundingCheck(BaseModel):
    """Checks if a generated answer is grounded in the context."""
    is_grounded : Literal["grounded", "not_grounded"] = Field(
        description=(
            "'grounded' if every factual claim in the answer can be "
            "verified from the context. 'not_grounded' if the answer "
            "contains unsupported claims. "
            "'I don't have enough information' is always 'grounded'."
        )
    )

class BatchDocumentGrades(BaseModel):
    """Grades for a batch of documents."""
    grades: List[Literal["yes", "no"]] = Field(
        description="One grade per document, in the same order as input"
    )

# ───────────────────────────────────────────
# AGENT STATE
# ───────────────────────────────────────────
 
class PolicyAgentState(TypedDict):
    """
    State for the Policy Agent.
 
    Uses TypedDict (not Pydantic) for LangGraph compatibility
    with add_messages annotation. Structured outputs above
    are Pydantic for the LLM boundary.
    """
 
    # ── Conversation ──
    # add_messages annotation auto-appends instead of replacing
    messages: Annotated[Sequence[BaseMessage], add_messages]
 
    # ── Current turn input ──
    question: str
 
    # ── Routing ──
    question_category: str  # "CHAT" or "RAG"
 
    # ── Query decomposition ──
    sub_queries: List[str]
 
    # ── Retrieval ──
    documents: List[Document]
 
    # ── Grading ──
    graded_documents: List[Document]
    grade_summary: str  # "all_relevant", "some_relevant", "none_relevant"
 
    # ── Generation ──
    answer: str
 
    # ── Grounding check ──
    is_grounded: bool
 
    # ── Retry counters ──
    retrieval_retry_count: int
    generation_retry_count: int