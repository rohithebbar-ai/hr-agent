"""
Policy Agent State
──────────────────
Pydantic BaseModel that flows between all LangGraph nodes.
Each node reads from state and returns partial updates.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class PolicyAgentState(BaseModel):
    """
    State that flows between LangGraph nodes.
 
    Each node receives the current state, does its work,
    and returns a dict with the fields it updated.
    LangGraph merges those updates into the state.
    """

    # -- input ---
    query: str = Field(
        description="Original User question"
    )

    # --- Routing ---
    route: str = Field(
        default = "",
        description="'hr_question' or 'out_of_scope"
    )

    # -- Query transformation --
    sub_queries: List[str] = Field(
        default_factory=list,
        description=(
            "Decomposed queries for multi-hop. "
            "If original query is simple, contains one item "
            "(the original query itself)."
        ),
    )

    # --- retrieval ---
    documents: List[Document] = Field(
        default_factory=list,
        description="All retrieved documents (merged from sub-queries)",
    )

    # -- Grading ---
    graded_documents: List[Document] = Field(
        default_factory=list,
        description="Documents that passed the relevance grader",
    )

    grade_summary: str = Field(
        default = "",
        description="'all_relevant', 'some_relevant', 'none_relevant'",
    )

    # -- Generation ---
    generation: str = Field(
        default="",
        description="Final answer from the generator",
    )
 
    # ── Grounding check ──
    is_grounded: bool = Field(
        default=True,
        description="Did the grounding check pass?",
    )

    # ── Control flow (retry counters) ──
    retrieval_retry_count: int = Field(
        default=0,
        description="How many times we've re-queried",
    )
    generation_retry_count: int = Field(
        default=0,
        description="How many times we've regenerated",
    )
 
    # ── Config ──
    class Config:
        arbitrary_types_allowed = True  # For Document type
 