"""
Policy-Aware Ingestion
──────────────────────
Ingests HR policies with structure awareness:
- Short policies embedded as complete units (no fragmentation)
- Long policies chunked with full policy metadata preserved
 
Writes to collection: hr_policy_aware
 
Usage:
    docker compose up -d
    uv run python -m rag.ingest_policy_aware
"""
from dotenv import load_dotenv
load_dotenv()
import sys
sys.stdout.flush()

import argparse
import json
from typing import Any

from langchain_core.documents import Document

from rag.config import (
    assert_policies_json_exists,
    ensure_recreate_allowed_if_production,
    policy_aware_policies_path,
    qdrant_recreate_from_env,
    resolve_qdrant_force_recreate,
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Policies shorter than this are embedded as a single unit.
# This prevents short policies from being split across chunks, which destroys context for the LLM.
WHOLE_POLICY_THRESHOLD = 1000

def load_policies() -> list[dict]:
    """ Load structured policy objects from preprocessing output """
    path = policy_aware_policies_path()
    assert_policies_json_exists(path)

    with open(path, "r", encoding="utf-8") as f:
        policies = json.load(f)

    print(f"[OK] Loaded {len(policies)} policies from {path.name}")
    return policies

def _build_metadata(policy: dict, doc_type: str, chunk_index: int = 0, chunk_total: int = 1) -> dict:
    """
    Build metadata dict for a document.
 
    Every document — whether a whole policy or a chunk — carries the full policy context. 
    """
    return {
        "policy_id": policy["policy_id"],
        "policy_name": policy["policy_name"],
        "section": policy["section"],
        "category": policy["category"],
        "keywords": policy.get("keywords", []),
        "page_start": policy["page_start"],
        "page_end": policy["page_end"],
        "doc_type": doc_type,
        "chunk_index": chunk_index,
        "chunk_total": chunk_total,
        "source": "VanaciPrime Employee Handbook",
    }

def create_documents(policies: list[dict]) -> list[Document]:
    """
    Convert policies into LangChain Documents.
 
    Strategy:
        len(text) <= 1000 chars → embed whole (doc_type: full_policy)
        len(text) >  1000 chars → chunk it   (doc_type: policy_chunk)
 
    Why 1000?
        all-MiniLM-L6-v2 has a max sequence length of 256 tokens (~1000 chars).
        Texts longer than this get truncated during embedding, losing information.
        Chunking ensures every piece of text fits within the model's window.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators = ["\n\n", "\n", ". ", " ", ""],
        length_function = len,
    )

    documents: list[Document] = []
    stats = {"whole": 0, "chunked": 0, "total_chunks" : 0}

    for policy in policies:
        text = policy["full_text"].strip()

        if not text:
            continue

        if len(text) <= WHOLE_POLICY_THRESHOLD:
            # --- Short policy -> single document ---
            documents.append(Document(
                page_content = text,
                metadata = _build_metadata(policy, doc_type="full_policy"),
            ))
            stats["whole"] += 1
        
        else:
            # ---- Long policy -> chunked with metadata ----
            chunks = splitter.split_text(text)
            stats["chunked"] += 1
            stats["total_chunks"] += len(chunks)

            for i , chunk_text in enumerate(chunks):
                documents.append(Document(
                    page_content = chunk_text,
                    metadata = _build_metadata(
                        policy,
                        doc_type="policy_chunk",
                        chunk_index=i,
                        chunk_total=len(chunks),
                    ),
                ))
    print(f"[OK] Created {len(documents)} documents:")
    print(f"Full policies [no_split]: {stats['whole']}")
    print(f"Chunked policies: {stats['chunked']} - {stats['total_chunks']} chunks")

    return documents


def ingest_to_qdrant(documents: list[Document], *, force_recreate: bool) -> Any:
    """ Embed and store in Qdrant """
    from langchain_qdrant import QdrantVectorStore

    from rag.retriever import COLLECTION_POLICY_AWARE, QDRANT_URL, get_embeddings

    embeddings = get_embeddings()

    print(
        f"[INGEST] Embedding {len(documents)} documents into '{COLLECTION_POLICY_AWARE}'.",
        flush=True,
    )

    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_POLICY_AWARE,
        force_recreate=force_recreate,
    )

    print(f"[OK] Ingested into: {COLLECTION_POLICY_AWARE}", flush=True)
    return vector_store

def verify_retrieval(vector_store: Any):
    """Smoke test — run a few queries and print results."""
    print("\n[VERIFY] Test queries (MMR, k=3):\n")
 
    queries = [
        "What is the sick leave policy?",
        "How many vacation days do employees get?",
        "What is the drug testing policy?",
        "What happens if I get injured at work?",
        "What is the overtime pay rate?",
    ]

    retriever = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs = {'k': 3},
    )

    for query in queries:
        results = retriever.invoke(query)
        if results:
            r = results[0]
            name = r.metadata.get("policy_name", "?")
            dtype = r.metadata.get("doc_type", "?")
            preview = r.page_content[:80].replace("\n", " ")
            print(f"Query:{query}")
            print(f"-> [{name}] ({dtype}) {preview}")
            print(f"{len(results)} results returned\n")
        else:
            print(f"Query: {query}\n -> No results\n")
            
def main():
    parser = argparse.ArgumentParser(description="Policy-aware HR RAG ingest")
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
    print("  Policy-Aware Ingestion", flush=True)
    print("=" * 60 + "\n", flush=True)

    policies = load_policies()
    documents = create_documents(policies)
    vector_store = ingest_to_qdrant(documents, force_recreate=force_recreate)
    verify_retrieval(vector_store)

if __name__ == "__main__":
    main()


