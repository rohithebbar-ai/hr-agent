"""
Policy Agent Nodes
──────────────────
Each node takes (state, **injected_deps) and returns a dict of state updates.
Dependencies (LLM, retriever) are injected via functools.partial in the pipeline builder.

Key change: try/except blocks removed from transform_query and check_grounding_node
so LLMWithFallback can catch rate limit errors and automatically retry with Groq.
"""

from typing import List, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END
from langgraph.types import Command
from agents.prompt_loader import load_prompt

from agents.schemas import (
    BatchDocumentGrades,
    GroundingCheck,
    PolicyAgentState,
    QueryDecomposition,
    RouteQuery,
)

# ── Dynamic behavior prompt (set via Admin panel → Redis) ─────────────────────
_DEFAULT_BEHAVIOR = "You are a helpful HR assistant for VanaciPrime."


def get_behavior_prompt() -> str:
    """Read the admin-configurable behavior prompt from Redis. Falls back to default."""
    try:
        from api.redis_client import is_redis_available, get_redis
        if is_redis_available():
            val = get_redis().get("hrassistant:behavior_prompt")
            if val:
                return val if isinstance(val, str) else val.decode()
    except Exception:
        pass
    return _DEFAULT_BEHAVIOR


def _extract_text(content) -> str:
    """Extract text from LLM response content (handles Gemini's list format)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block["text"]
            if isinstance(block, str):
                return block
        return str(content)
    return str(content)


# ══════════════════════════════════════════════════
# NODE 1: route_query
# ══════════════════════════════════════════════════

ROUTER_PROMPT = load_prompt("router", version="v1")


def route_query_node(
    state: PolicyAgentState,
    base_llm,
) -> Command[Literal["chat_node", "transform_query"]]:
    """Classify query as CHAT or RAG and route accordingly."""
    print(f"\n[ROUTE] Classifying: {state['question'][:60]}")

    try:
        structured_llm = base_llm.with_structured_output(RouteQuery)
        chain = ROUTER_PROMPT | structured_llm
        result = chain.invoke({"question": state["question"]})
        category = result.category
    except Exception as e:
        print(f"[ROUTE] Routing failed: {e}, defaulting to CHAT")
        category = "CHAT"

    goto = "chat_node" if category == "CHAT" else "transform_query"
    print(f"[ROUTE] → {category} → {goto}")

    return Command(
        update={
            "question_category": category,
            "messages": [HumanMessage(content=state["question"])],
        },
        goto=goto,
    )


# ══════════════════════════════════════════════════
# NODE 2: chat_node
# ══════════════════════════════════════════════════

CHAT_PROMPT = load_prompt("chat", version="v1")


def chat_node(
    state: PolicyAgentState,
    base_llm,
) -> Command[Literal["__end__"]]:
    """Handle conversational questions without retrieval."""
    print("[CHAT] Direct chat response (no retrieval)")

    behavior = get_behavior_prompt()
    prompt = CHAT_PROMPT.partial(behavior=behavior)
    chain = prompt | base_llm
    history = state.get("messages", [])[-6:]

    try:
        response = chain.invoke({
            "question": state["question"],
            "history": history,
        })
        answer = _extract_text(response.content)
    except Exception as e:
        print(f"[CHAT] LLM failed: {e}")
        answer = (
            "I'm sorry, but I can only help with HR-related "
            "questions. If you have any questions about company "
            "policies, benefits, or procedures, feel free to ask!"
        )

    print(f"[CHAT] → Generated {len(answer)} chars")

    return Command(
        update={
            "answer": answer,
            "messages": [AIMessage(content=answer)],
        },
        goto=END,
    )


# ══════════════════════════════════════════════════
# NODE 3: transform_query
# ══════════════════════════════════════════════════

TRANSFORM_PROMPT = load_prompt("transform", version="v1")


def transform_query(
    state: PolicyAgentState,
    base_llm,
) -> Command[Literal["retrieve"]]:
    """
    Decompose multi-hop queries into focused sub-queries.

    No try/except here — LLMWithFallback handles rate limit errors
    by automatically retrying with Groq when Gemini quota is exhausted.
    If the LLM fails for any other reason, the exception propagates
    and FastAPI returns a 500 (correct behavior for unexpected failures).
    """
    print("\n[TRANSFORM] Analyzing query for decomposition")

    structured_llm = base_llm.with_structured_output(QueryDecomposition)
    chain = TRANSFORM_PROMPT | structured_llm
    history = state.get("messages", [])[-6:]

    # LLMWithFallback will catch 429/quota errors and retry with Groq.
    # No manual fallback needed here.
    result = chain.invoke({
        "question": state["question"],
        "history": history,
    })
    sub_queries = result.sub_queries

    # Defensive: ensure we always have at least the original question
    if not sub_queries:
        sub_queries = [state["question"]]

    print(f"[TRANSFORM] → {len(sub_queries)} sub-queries:")
    for i, q in enumerate(sub_queries, 1):
        print(f"{i}. {q}")

    return Command(
        update={"sub_queries": sub_queries},
        goto="retrieve",
    )


# ══════════════════════════════════════════════════
# NODE 4: retrieve
# ══════════════════════════════════════════════════

def retrieve_node(state: PolicyAgentState) -> Command:
    """Retrieve using enterprise hybrid retrieval + reranker."""
    sub_queries = state.get("sub_queries", [state["question"]])
    all_docs = []

    for query in sub_queries:
        from rag.retriever_enterprise import retrieve
        docs = retrieve(
            query=query,
            tenant_id="vanaciprime",
            k=8,
        )
        all_docs.extend(docs)

    # Deduplicate by chunk_id
    seen = set()
    unique_docs = []
    for doc in all_docs:
        cid = doc.metadata.get("chunk_id")
        if cid not in seen:
            seen.add(cid)
            unique_docs.append(doc)

    print(f"[RETRIEVE] {len(unique_docs)} unique docs from hybrid search")

    return Command(
        update={"documents": unique_docs},
        goto="grade_documents",
    )


# ══════════════════════════════════════════════════
# NODE 5: grade_documents (BATCHED)
# ══════════════════════════════════════════════════

BATCH_GRADER_PROMPT = load_prompt("grader", version="v1")


def grade_documents_node(
    state: PolicyAgentState,
    base_llm,
) -> Command[Literal["generate", "transform_query"]]:
    """
    Grade all documents in ONE batched LLM call.
    Uses section_path for display (matches enterprise retriever metadata).
    """
    documents = state["documents"]
    print(f"\n[GRADE] Batch-grading {len(documents)} documents")

    # Handle empty case
    if not documents:
        print("[GRADE] → No documents to grade")
        retry_count = state.get("retrieval_retry_count", 0)
        if retry_count < 2:
            print(f"[GRADE] → Retrying (attempt {retry_count + 1}/2)")
            return Command(
                update={
                    "graded_documents": [],
                    "grade_summary": "none_relevant",
                    "retrieval_retry_count": retry_count + 1,
                },
                goto="transform_query",
            )
        return Command(
            update={
                "graded_documents": [],
                "grade_summary": "none_relevant",
            },
            goto="generate",
        )

    def _truncate_for_grading(doc, max_chars=300):
        text = doc.page_content
        return text[:max_chars] + "..." if len(text) > max_chars else text

    # Use section_path for grading context (enterprise retriever metadata)
    doc_text = "\n\n".join([
        f"[Document {i+1}] "
        f"(Section: {doc.metadata.get('section_path', doc.metadata.get('section_title', 'Unknown'))[:60]})\n"
        f"{_truncate_for_grading(doc)}"
        for i, doc in enumerate(documents)
    ])

    structured_llm = base_llm.with_structured_output(BatchDocumentGrades)
    chain = BATCH_GRADER_PROMPT | structured_llm

    try:
        result = chain.invoke({
            "question": state["question"],
            "documents": doc_text,
        })
        grades = result.grades

        if len(grades) != len(documents):
            print(
                f"[GRADE] Grade count mismatch "
                f"({len(grades)} grades for {len(documents)} docs), "
                f"keeping all"
            )
            grades = ["yes"] * len(documents)

    except Exception as e:
        print(f"[GRADE] Batch grading failed: {e}, keeping all docs")
        grades = ["yes"] * len(documents)

    # Log per-document decisions with section_path
    graded = []
    for i, (doc, grade) in enumerate(zip(documents, grades)):
        label = doc.metadata.get(
            "section_path",
            doc.metadata.get("section_title", "Unknown")
        )[:50]
        marker = "correct" if grade == "yes" else "wrong"
        print(f"  {marker} [{grade}] {label}")
        if grade == "yes":
            graded.append(doc)

    total = len(documents)
    kept = len(graded)

    if kept == 0:
        summary = "none_relevant"
    elif kept == total:
        summary = "all_relevant"
    else:
        summary = "some_relevant"

    print(f"[GRADE] → {kept}/{total} relevant ({summary})")

    # Safety net: if grader too aggressive, keep all
    if total >= 4 and kept / total < 0.25:
        print(
            f"[GRADE] → Grader too aggressive "
            f"({kept}/{total} = {kept/total:.0%}), "
            f"keeping all docs as safety net"
        )
        graded = documents
        summary = "all_relevant"
        kept = total

    # If grader rejected everything (suspicious), keep all
    if kept == 0 and total > 0:
        print("[GRADE] → Grader rejected all docs, keeping all as safety net")
        return Command(
            update={
                "graded_documents": documents,
                "grade_summary": "all_relevant",
            },
            goto="generate",
        )

    retry_count = state.get("retrieval_retry_count", 0)

    if summary == "none_relevant" and retry_count < 2:
        print(f"[GRADE] → Retrying retrieval (attempt {retry_count + 1}/2)")
        return Command(
            update={
                "graded_documents": graded,
                "grade_summary": summary,
                "retrieval_retry_count": retry_count + 1,
            },
            goto="transform_query",
        )

    return Command(
        update={
            "graded_documents": graded,
            "grade_summary": summary,
        },
        goto="generate",
    )


# ══════════════════════════════════════════════════
# NODE 6: generate
# ══════════════════════════════════════════════════

GENERATE_PROMPT = load_prompt("generate", version="v1")


def _format_context(docs: List[Document]) -> str:
    """
    Format documents into context string for the LLM.
    Uses section_path from enterprise retriever metadata.
    """
    if not docs:
        return "(no documents retrieved)"

    formatted = []
    for i, doc in enumerate(docs, 1):
        # Enterprise retriever stores section_path, not policy_name/section
        section = doc.metadata.get(
            "section_path",
            doc.metadata.get("section_title",
            doc.metadata.get("section", "Unknown"))
        )
        formatted.append(
            f"[Source {i} | Section: {section}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(formatted)


def generate_node(
    state: PolicyAgentState,
    base_llm,
) -> Command[Literal["check_grounding"]]:
    """Generate answer from graded documents."""
    print(f"\n[GENERATE] Using {len(state['graded_documents'])} documents...")

    context = _format_context(state["graded_documents"])
    history = state.get("messages", [])[-6:]

    behavior = get_behavior_prompt()
    prompt = GENERATE_PROMPT.partial(behavior=behavior)
    chain = prompt | base_llm

    response = chain.invoke({
        "question": state["question"],
        "context": context,
        "history": history,
    })
    answer = _extract_text(response.content)
    print(f"[GENERATE] → Generated {len(answer)} chars")
    print(f"[GENERATE] → Preview: {answer[:80]}...")

    return Command(
        update={"answer": answer},
        goto="check_grounding",
    )


# ══════════════════════════════════════════════════
# NODE 7: check_grounding
# ══════════════════════════════════════════════════

GROUNDING_PROMPT = load_prompt("grounding", version="v1")


def check_grounding_node(
    state: PolicyAgentState,
    base_llm,
) -> Command[Literal["generate", "__end__"]]:
    """
    Check if the answer is grounded in retrieved context.

    No try/except here — LLMWithFallback handles rate limit errors
    by automatically retrying with Groq when Gemini quota is exhausted.
    Short refusal answers are skipped to avoid unnecessary LLM calls.
    """
    answer = state["answer"]

    # Skip grounding check for clear refusal answers — saves LLM call
    refusal_phrases = [
        "don't have enough",
        "i don't have",
        "cannot answer",
        "not enough information",
    ]

    if len(answer) < 150 and any(
        phrase in answer.lower() for phrase in refusal_phrases
    ):
        print("\n[GROUND] Skipped (refusal detected)")
        return Command(
            update={
                "is_grounded": True,
                "messages": [AIMessage(content=answer)],
            },
            goto=END,
        )

    print("\n[GROUND] Checking grounding.")

    # Use top 5 documents, truncated for speed
    context_snippets = []
    for i, doc in enumerate(state["graded_documents"][:5], 1):
        snippet = doc.page_content[:500]
        context_snippets.append(f"[Doc {i}] {snippet}")
    context = "\n\n".join(context_snippets)

    structured_llm = base_llm.with_structured_output(GroundingCheck)
    chain = GROUNDING_PROMPT | structured_llm

    # LLMWithFallback will catch 429/quota errors and retry with Groq.
    # No manual fallback needed here.
    result = chain.invoke({
        "context": context,
        "answer": answer,
    })
    is_grounded = result.is_grounded == "grounded"

    print(f"[GROUND] → {'grounded' if is_grounded else 'NOT grounded'}")

    retry_count = state.get("generation_retry_count", 0)

    if not is_grounded and retry_count < 1:
        print(f"[GROUND] → Regenerating (attempt {retry_count + 1}/1)")
        return Command(
            update={
                "is_grounded": is_grounded,
                "generation_retry_count": retry_count + 1,
            },
            goto="generate",
        )

    return Command(
        update={
            "is_grounded": is_grounded,
            "messages": [AIMessage(content=answer)],
        },
        goto=END,
    )