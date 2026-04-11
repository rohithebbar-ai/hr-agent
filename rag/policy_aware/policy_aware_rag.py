"""
Policy-Aware RAG Chain
──────────────────────
Uses policy-aware Qdrant collection with MMR retrieval.
Same prompt template as naive chain — only retrieval differs.
This isolates the impact of chunking strategy.
 
Usage:
    uv run python -m rag.chain_policy_aware
"""

import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from rag.retriever import (
    COLLECTION_POLICY_AWARE,
    get_rerank_retriever,
    get_retriever
)
from scripts.llm_manager import LLMTask, get_llm

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")

SYSTEM_PROMPT = """You are an HR Policy Assistant for VanaciPrime.
Answer the employee's question based ONLY on the provided context.
If the context does not contain enough information to answer,
say "I don't have enough information in our HR policies to
answer that question."
 
Always cite which policy or section your answer comes from.
 
Context:
{context}
"""

PROMPT = ChatPromptTemplate.from_messages([
    ('system', SYSTEM_PROMPT),
    ("human", "{question}"),
])

def format_docs(docs: list) -> str:
    """
    Format retrieved documents into context string.
 
    Includes policy name and doc_type so the LLM knows
    whether it's reading a complete policy or a chunk.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        policy = doc.metadata.get("policy_name", "Unknown")
        section = doc.metadata.get("section", "Unknown")
        doc_type = doc.metadata.get("doc_type", "unknown")
 
        header = (
            f"[Source {i} | "
            f"Policy: {policy} | "
            f"Section: {section} | "
            f"Type: {doc_type}]"
        )
        formatted.append(f"{header}\n{doc.page_content}")
 
    return "\n\n".join(formatted)

def build_chain():
    """
    Build RAG chain with policy-aware retrieval.
 
    Differences from naive chain:
    - Collection: hr_policy_aware (not hr_naive)
    - Search: MMR (not pure similarity)
    - Context: includes policy metadata in formatting
    """
    retriever = get_retriever(
        collection=COLLECTION_POLICY_AWARE,
        search_type="mmr",
        k=5,
    )

    #llm = ChatGroq(model=LLM_MODEL, temperature=0)
    llm = get_llm(LLMTask.GENERATION)

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )


def main():
    print("\n" + "=" * 60)
    print("  Policy-Aware RAG Chain (MMR retrieval)")
    print("  Type 'quit' to exit")
    print("=" * 60 + "\n")

    chain = build_chain()

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print("\nThinking...\n")
        print(chain.invoke(question))


if __name__ == "__main__":
    main()
