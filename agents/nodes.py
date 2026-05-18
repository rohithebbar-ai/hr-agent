"""
Policy Agent Nodes
──────────────────
Each node takes (state, **injected_deps) and returns a dict of state updates.
Dependencies (LLM, retriever) are injected via functools.partial in the pipeline builder — no globals.
"""

from typing import List, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from langgraph.types import Command
from agents.prompt_loader import load_prompt

from agents.schemas import(
    BatchDocumentGrades,
    GroundingCheck,
    PolicyAgentState,
    QueryDecomposition,
    RouteQuery,
)

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
# Classifies query as CHAT (small talk) or RAG (needs retrieval).


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
# Handles non-RAG questions directly without retrieval.

CHAT_PROMPT = CHAT_PROMPT = load_prompt("chat", version="v1")


def chat_node(
    state: PolicyAgentState,
    base_llm,
    ) -> Command[Literal["__end__"]]:
    """Handle conversational questions without retrieval"""
    print(f"[CHAT] Direct chat response (no retrieval)")

    chain = CHAT_PROMPT | base_llm
    history = state.get("messages", [])[-6:]

    try:
        response = chain.invoke({
            "question": state["question"],
            "history": history,
        })
        answer = _extract_text(response.content)
    except Exception as e:
        print(f"[CHAT] LLM failed: {e}")
        # Fallback response for non-HR questions
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
# Decomposes multi-hop queries into sub-queries.

TRANSFORM_PROMPT = load_prompt("transform", version="v1")

def transform_query(
    state: PolicyAgentState,
    base_llm,
) -> Command[Literal["retrieve"]]:
    """Decompose multi-hop queries into focused sub-queries """
    print(f"\n[TRANSFORM] Analyzing query for decomposition")

    structured_llm = base_llm.with_structured_output(QueryDecomposition)
    chain = TRANSFORM_PROMPT | structured_llm

    history = state.get("messages", [])[-6:]

    try:
        result = chain.invoke({
            "question": state["question"],
            "history": history,
        })
        sub_queries = result.sub_queries
    except Exception as e:
        print(f"[TRANSFORM] Decomposition failed: {e}")
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

def retrieve_node(
    state: PolicyAgentState,
    retriever,
)-> Command[Literal["grade_documents"]]:
    """
    Retrieve documents for each sub-query.
    Dedup by policy_id + chunk_index.
    """
    sub_queries = state.get("sub_queries") or [state["question"]]
    print(f"\n[RETRIEVE] Fetching for {len(sub_queries)} sub-queries")

    all_docs: List[Document] = []
    seen_keys = set()

    for sub_query in sub_queries:
        docs = retriever.invoke(sub_query)
        for doc in docs:
            key = (
                doc.metadata.get("policy_id", ""),
                doc.metadata.get("chunk_index", 0),
            )
            if key not in seen_keys:
                seen_keys.add(key)
                all_docs.append(doc)

    
    doc_ids = [d.metadata.get("policy_id", "?") for d in all_docs[:5]]
    print(f"[RETRIEVE] → {len(all_docs)} unique documents")
    print(f"[RETRIEVE] → Top policy IDs: {doc_ids}")

    return Command(
        update={"documents": all_docs},
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

    Routes to retry or generate based on grading result.
    Uses a generous grading prompt to avoid filtering out
    relevant documents.
    """
    documents = state["documents"]
    print(f"\n[GRADE] Batch-grading {len(documents)} documents")

    # ── Handle empty case ──
    if not documents:
        print(f"[GRADE] → No documents to grade")
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

    # ── Format documents for batch grading ──
    def _truncate_for_grading(doc, max_chars=300):
        """Truncate document text for grading — grader only needs enough to judge relevance."""
        text = doc.page_content
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text

    doc_text = "\n\n".join([
        f"[Document {i+1}] (Policy: {doc.metadata.get('policy_name', 'Unknown')})\n"
        f"{_truncate_for_grading(doc)}"
        for i, doc in enumerate(documents)
    ])

    # ── Run batched grader ──
    structured_llm = base_llm.with_structured_output(BatchDocumentGrades)
    chain = BATCH_GRADER_PROMPT | structured_llm

    try:
        result = chain.invoke({
            "question": state["question"],
            "documents": doc_text,
        })
        grades = result.grades

        # Defensive: handle mismatched grade count
        if len(grades) != len(documents):
            print(
                f"[GRADE] Grade count mismatch "
                f"({len(grades)} grades for {len(documents)} docs), "
                f"keeping all documents"
            )
            grades = ["yes"] * len(documents)

    except Exception as e:
        print(f"[GRADE] Batch grading failed: {e}, keeping all docs")
        grades = ["yes"] * len(documents)

    # ── Filter and log per-document decisions ──
    graded = []
    for i, (doc, grade) in enumerate(zip(documents, grades)):
        policy = doc.metadata.get("policy_name", "?")[:50]
        marker = "correct" if grade == "yes" else "wrong"
        print(f"  {marker} [{grade}] {policy}")
        if grade == "yes":
            graded.append(doc)

    # ── Calculate summary ──
    total = len(documents)
    kept = len(graded)

    if kept == 0:
        summary = "none_relevant"
    elif kept == total:
        summary = "all_relevant"
    else:
        summary = "some_relevant"

    print(f"[GRADE] → {kept}/{total} relevant ({summary})")
    if total >= 4 and kept / total < 0.25:
        print(
            f"[GRADE] → Grader too aggressive "
            f"({kept}/{total} = {kept/total:.0%}), "
            f"keeping all docs as safety net"
        )
        graded = documents
        summary = "all_relevant"
        kept = total

    # ── Safety net: if grader rejected EVERYTHING but we have docs,
    # ── that's almost certainly a grader failure. Keep all docs anyway.
    if kept == 0 and total > 0:
        print(f"[GRADE] → Grader rejected all docs (suspicious), "
              f"keeping all as safety net")
        return Command(
            update={
                "graded_documents": documents,  # Keep ALL
                "grade_summary": "all_relevant",
            },
            goto="generate",
        )

    # ── Decide routing ──
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
    """Format documents into context string for the LLM."""
    if not docs:
        return "(no documents retrieved)"

    formatted = []
    for i, doc in enumerate(docs, 1):
        policy = doc.metadata.get("policy_name", "Unknown")
        section = doc.metadata.get("section", "Unknown")
        formatted.append(
            f"[Source {i} | Policy: {policy} | Section: {section}]\n"
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

    chain = GENERATE_PROMPT | base_llm
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
    """Check if the answer is grounded. Retry generation if not."""
    answer = state["answer"]

    # OPTIMIZATION: skip grounding for refusal answers
    refusal_phrases = [
        "don't have enough",
        "i don't have",
        "cannot answer",
        "not enough information",
    ]

    if (len(answer) < 150 and
            any(phrase in answer.lower() for phrase in refusal_phrases)):
        print(f"\n[GROUND] Skipped (refusal detected)")
        return Command(
            update={
                "is_grounded": True,
                "messages": [AIMessage(content=answer)],
            },
            goto=END,
        )

    print(f"\n[GROUND] Checking grounding.")

    context_snippets = []
    for i, doc in enumerate(state["graded_documents"][:5], 1):
        snippet = doc.page_content[:500]
        context_snippets.append(f"[Doc {i}] {snippet}")

    context = "\n\n".join(context_snippets)
    
    structured_llm = base_llm.with_structured_output(GroundingCheck)
    chain = GROUNDING_PROMPT | structured_llm
    try:
        result = chain.invoke({
            "context": context,
            "answer": answer,
        })
        is_grounded = result.is_grounded == "grounded"
    except Exception as e:
        print(f"[GROUND] Check failed: {e}, assuming grounded")
        is_grounded = True

    print(f"[GROUND] → {'grounded' if is_grounded else 'NOT grounded'}")

    # Decide routing inline
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







