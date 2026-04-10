"""
Naive RAG Chain (Baseline)

Usage:
    uv run python rag/baseline_rag.py
"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from rag.retriever import get_retriever

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

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
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

def format_docs(docs: list) -> str:
    """Format retrieved documents into a context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        policy = doc.metadata.get("policy_name", "Unknown")
        section = doc.metadata.get("section", "Unknown")
        formatted.append(
            f"[Source {i} | Policy: {policy} | "
            f"Section: {section}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(formatted)

def baseline_rag_chain():
    """
    Build baseline RAG chain.
    Pure similarity search, k = 5 
    """
    # Baseline: similarity search
    retriever = get_retriever(search_type="similarity", k=5)

    llm = ChatGroq(
        model = LLM_MODEL,
        temperature=0.5,
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return chain

def main():
    print("\n" + "=" * 60)
    print("  Naive RAG Chain (Baseline)")
    print("=" * 60 + "\n")

    chain = baseline_rag_chain()

    while True:
        question = input("\nYour Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break

        if not question:
            continue

        print("\nThinking...\n")
        response = chain.invoke(question)
        print(f"Answer: {response}")


if __name__ == "__main__":
    main()