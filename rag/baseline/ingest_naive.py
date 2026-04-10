"""
RAG Pipeline - Baseline Ingestion
──────────────────────────────────────────
Loads policies.json, does BLIND 500-token chunking (baseline),
embeds with all-MiniLM-L6-v2, stores in Qdrant.

Baseline = simplest version. No policy-aware chunking yet.

Usage:
    docker compose up -d
    uv run python -m rag.ingest_naive
"""

import argparse
import json

from typing import Any

from dotenv import load_dotenv

print("[START] Loading LangChain (light); PyTorch loads only when embedding…", flush=True)
from langchain_core.documents import Document
from rag.config import (
    assert_policies_json_exists,
    ensure_recreate_allowed_if_production,
    naive_policies_path,
    qdrant_recreate_from_env,
    resolve_qdrant_force_recreate,
)

load_dotenv()

# Baseline chunking parameters

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_policies() -> list[Document]:
    """ Load policies.json into Langchain Documents"""
    policies_path = naive_policies_path()
    assert_policies_json_exists(policies_path)

    with open(policies_path, "r", encoding="utf-8") as f:
        policies = json.load(f)

    documents = []
    for policy in policies:
        documents.append(Document(
            page_content=policy["full_text"],
            metadata ={
                "policy_id": policy["policy_id"],
                "policy_name": policy["policy_name"],
                "section": policy["section"],
                "category": policy["category"],
                "page_start": policy["page_start"],
                "source": "VanaciPrime Employee Handbook",
            }
        ))
    print(f"[OK] Loaded {len(documents)} policies from policies.json")
    return documents

def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Baseline chunking using recursive splitter
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ",""],
        length_function = len,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        print("[OK] Created 0 chunks (no content to chunk)")
        return chunks

    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_total"] = len(chunks)

    lengths = [len(c.page_content) for c in chunks]
    print(f"[OK] Created {len(chunks)} chunks")
    print(f"Avg: {sum(lengths) // len(lengths)} chars")
    print(f"Min: {min(lengths)}, Max: {max(lengths)} chars")

    return chunks

def ingest_to_qdrant(
    chunks: list[Document],
    embeddings,
    *,
    force_recreate: bool,
) -> Any:
    """Embed chunks and store in Qdrant"""
    from langchain_qdrant import QdrantVectorStore

    from rag.retriever import COLLECTION_NAIVE, QDRANT_URL

    print(f"[INGEST] Embedding {len(chunks)} chunks", flush=True)

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding = embeddings,
        url = QDRANT_URL,
        collection_name = COLLECTION_NAIVE,
        force_recreate = force_recreate,
    )
    print(f"[OK] Ingested into Qdrant: {COLLECTION_NAIVE}", flush=True)
    return vector_store


def verify_retrieval(vector_store: Any):
    """Quick test queries to verify retrieval works."""
    print("\n[TEST] Running test queries...\n")

    test_queries = [
        "What is the sick leave policy?",
        "How many vacation days do employees get?",
        "What is the drug testing policy?",
        "How does health insurance work?",
        "What is the overtime pay rate?",
    ]

    # Baseline
    retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": 3},
    )

    for query in test_queries:
        print(f"Q: {query}")
        results = retriever.invoke(query)
        if results:
            preview = results[0].page_content[:100].replace("\n", " ")
            policy = results[0].metadata.get("policy_name", "?")
            print(f"A:[{policy}]{preview}")
            print(f"({len(results)} chunks retrieved)")
        else:
            print("A: No results found")
        print()


def main():
    parser = argparse.ArgumentParser(description="Baseline HR RAG ingest")
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Drop and recreate the Qdrant collection (also via QDRANT_RECREATE_COLLECTION=true).",
    )
    args = parser.parse_args()

    env_recreate = qdrant_recreate_from_env()
    force_recreate = resolve_qdrant_force_recreate(
        env_flag=env_recreate,
        cli_recreate=args.recreate_collection,
    )
    ensure_recreate_allowed_if_production(force_recreate)

    print("\n" + "=" * 60, flush=True)
    print("Baseline RAG Pipeline", flush=True)
    print("=" * 60 + "\n", flush=True)

    # Load and chunk first (fast); PyTorch/sentence-transformers load only when embedding.
    documents = load_policies()
    chunks = chunk_documents(documents)

    from rag.retriever import get_embeddings

    embeddings = get_embeddings()

    vector_store = ingest_to_qdrant(chunks, embeddings, force_recreate=force_recreate)

    # Test
    verify_retrieval(vector_store)


if __name__ == "__main__":
    main()


