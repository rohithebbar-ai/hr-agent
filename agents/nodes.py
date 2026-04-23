"""
Policy Agent Nodes
──────────────────
Each node takes (state, **injected_deps) and returns a dict of state updates.
Dependencies (LLM, retriever) are injected via functools.partial in the pipeline builder — no globals.

functools.partial is a powerful tool in the Python Standard Library used for partial function application. It allows you to "freeze" a portion of a function's arguments or keywords, creating a new callable object with a simplified signature

Example
from functools import partial

def multiply(x, y):
    return x * y

# Create a 'double' function by pre-filling x = 2
double = partial(multiply, 2)

print(double(4))  # Output: 8 (Equivalent to multiply(2, 4))
"""

from typing import List, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from langgraph.types import Command

from agents.schemas import(
    BatchDocumentGrades,
    GroundingCheck,
    PolicyAgentState,
    QueryDecomposition,
    RouteQuery,
)

# ══════════════════════════════════════════════════
# NODE 1: route_query
# ══════════════════════════════════════════════════
# Classifies query as CHAT (small talk) or RAG (needs retrieval).


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query router for VanaciPrime's HR assistant.\n"
     "Classify the user's question:\n"
     "- 'RAG' if it requires HR policy information, benefits, "
     "leave details, workplace conduct, compensation, or any "
     "topic from an employee handbook\n"
     "- 'CHAT' for greetings, thanks, small talk, meta-questions "
     "about the assistant, or general questions not requiring "
     "document retrieval"),
    ("human", "{question}"),
])

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

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the HR Policy Assistant for VanaciPrime. "
     "You are friendly and helpful. For greetings, small talk, "
     "or meta questions, respond briefly and naturally.\n\n"
     "If the user asks about a topic outside HR (like coding, "
     "weather, general knowledge), politely explain that you "
     "only help with HR policies and suggest they ask an HR "
     "question instead.\n\n"
     "Keep responses under 3 sentences."),
    ("placeholder", "{history}"),
    ("human", "{question}"),
])


def chat_node(
    state: PolicyAgentState,
    base_llm,
    ) -> Command[Literal["__end__"]]:
    """Handle conversational questions without retrieval"""
    print(f"[CHAT] Direct chat response (no retrieval)")

    chain = CHAT_PROMPT | base_llm
    history = state.get("messages", [])[-6:]

    response = chain.invoke({
        "question": state["question"],
        "history": history,
    })

    answer = response.content
    print(f"[CHAT] -> Generated {len(answer)} chars")

    return Command(
        update={
            "answer": answer,
            "messages": [response],
        },
        goto=END,
    )

# ══════════════════════════════════════════════════
# NODE 3: transform_query
# ══════════════════════════════════════════════════
# Decomposes multi-hop queries into sub-queries.

TRANSFORM_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You help rewrite HR questions for better document retrieval.\n\n"
     "Analyze the user's question:\n"
     "- If it's a simple single-topic question, return it as one query.\n"
     "- If it's a comparison, multi-hop, or conditional question "
     "that needs information from multiple policies, decompose "
     "into 2-4 focused sub-queries.\n\n"
     "Use conversation history to resolve pronouns and references.\n\n"
     "Examples:\n"
     "'How many vacation days?' → ['How many vacation days?']\n"
     "'How does FMLA differ from personal leave?' → "
     "['What is the FMLA leave policy?', "
     "'What is the personal leave policy?']\n"
     "'If I exhaust PTO and sick leave, what options remain?' → "
     "['What is the PTO policy?', "
     "'What is the sick leave policy?', "
     "'What unpaid leave options exist?']"),
    ("placeholder", "{history}"),
    ("human", "{question}"),
])

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

BATCH_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a relevance grader for an HR retrieval system.\n"
     "For each document, determine if it could help answer the "
     "user's question — even partially.\n\n"
     "Be GENEROUS with relevance. A document is relevant if:\n"
     "- It mentions the topic in the question\n"
     "- It contains policies related to the topic\n"
     "- It provides background or context for the answer\n"
     "- It defines terms used in the question\n"
     "- It's from the same policy area (e.g., 'Vacation Policy' "
     "is relevant to 'How many vacation days?')\n\n"
     "A document is NOT relevant only if it's about a "
     "completely different topic with no connection to the question.\n\n"
     "When in doubt, mark as 'yes'. Better to keep too much "
     "context than to lose relevant information.\n\n"
     "Return one grade per document in the same order."),
    ("human",
     "Question: {question}\n\n"
     "Documents:\n{documents}\n\n"
     "Grade each document as 'yes' or 'no'."),
])


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
    print(f"\n[GRADE] Batch-grading {len(documents)} documents...")

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
    doc_text = "\n\n".join([
        f"[Document {i+1}] (Policy: {doc.metadata.get('policy_name', 'Unknown')})\n"
        f"{doc.page_content[:500]}"
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


GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an HR Policy Assistant for VanaciPrime.\n"
     "Answer the employee's question based ONLY on the provided "
     "context. If the context doesn't contain enough information, "
     "say so clearly.\n\n"
     "IMPORTANT formatting rules:\n"
     "- Do NOT include source numbers or citations inline "
     "(no 'Source 1', 'Source 2', etc.)\n"
     "- Do NOT mention policy sections or document metadata "
     "in your answer\n"
     "- Just answer naturally as if you know the information\n"
     "- Keep answers concise and friendly\n\n"
     "Context:\n{context}"),
    ("placeholder", "{history}"),
    ("human", "{question}"),
])

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
    answer = response.content
    print(f"[GENERATE] → Generated {len(answer)} chars")
    print(f"[GENERATE] → Preview: {answer[:80]}...")

    return Command(
        update={"answer": answer},
        goto="check_grounding",
    )

# ══════════════════════════════════════════════════
# NODE 7: check_grounding
# ══════════════════════════════════════════════════


GROUNDING_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a hallucination checker. Determine if the given "
     "answer is fully grounded in the provided context.\n\n"
     "An answer is GROUNDED if every factual claim can be "
     "verified from the context.\n"
     "An answer is NOT GROUNDED if it contains claims not "
     "supported by the context.\n\n"
     "'I don't have enough information' answers are always GROUNDED."),
    ("human",
     "Context:\n{context}\n\n"
     "Answer:\n{answer}\n\n"
     "Is the answer grounded?"),
])

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

    print(f"\n[GROUND] Checking grounding...")

    context_snippets = []
    for i, doc in enumerate(state["graded_documents"][:5], 1):  # Max 5 docs
        snippet = doc.page_content[:300]  # Truncate
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







