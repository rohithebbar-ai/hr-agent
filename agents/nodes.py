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

    structured_llm = base_llm.with_structured_output(RouteQuery)
    chain = ROUTER_PROMPT | structured_llm

    result = chain.invoke({"question": state["question"]})
    category = result.category

    goto = "chat_node" if category == "CHAT" else "transform_query"
    print(f"[ROUTE] -> {category} -> {goto}")

    return Command(
        update={
            "question_category": category,
            "messages": [HumanMessage(content=state["question"])]
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

