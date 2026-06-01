# rag/retriever_enterprise.py

import os
from dotenv import load_dotenv
from typing import List

load_dotenv()
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    SparseVector, Prefetch, FusionQuery, Fusion,
)
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest

COLLECTION_NAME = "hr_documents"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
CANDIDATES = 40
FINAL_K = 8

_dense_model = _sparse_model = _ranker = _client = None


def get_dense_model():
    global _dense_model
    if _dense_model is None:
        _dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _dense_model


def get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name="prithvida/Splade_PP_en_v1")
    return _sparse_model


def get_ranker():
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return _ranker


def get_client():
    global _client
    if _client is None:
        url = os.environ.get("QDRANT_URL")
        api_key = os.environ.get("QDRANT_API_KEY")
        _client = QdrantClient(url=url, api_key=api_key, timeout=30) if api_key else QdrantClient(url=url, timeout=30)
    return _client


def retrieve(query: str, tenant_id: str = "vanaciprime", k: int = FINAL_K, debug: bool = False) -> List[Document]:
    """
    Hybrid retrieval: dense + sparse → Qdrant RRF → rerank → top-k.
    """
    dense_vec = get_dense_model().encode(query, normalize_embeddings=True).tolist()

    sparse_result = list(get_sparse_model().embed([query]))[0]
    sparse_vec = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    tenant_filter = Filter(must=[
        FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
    ])

    client = get_client()

    if debug:
        dense_only = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[Prefetch(query=dense_vec, using=DENSE_VECTOR_NAME, filter=tenant_filter, limit=CANDIDATES)],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=CANDIDATES,
            with_payload=True,
        )
        sparse_only = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[Prefetch(query=sparse_vec, using=SPARSE_VECTOR_NAME, filter=tenant_filter, limit=CANDIDATES)],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=CANDIDATES,
            with_payload=True,
        )
        dense_ids = {p.id for p in dense_only.points}
        sparse_ids = {p.id for p in sparse_only.points}
        print(f"[DEBUG] dense-only candidates: {len(dense_ids)}")
        print(f"[DEBUG] sparse-only candidates: {len(sparse_ids)}")
        print(f"[DEBUG] sparse-unique (not in dense): {len(sparse_ids - dense_ids)}")
        for p in sparse_only.points:
            if p.id not in dense_ids:
                print(f"  [SPARSE-ONLY] {p.payload.get('section_path', '')[:60]}")

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec, using=DENSE_VECTOR_NAME, filter=tenant_filter, limit=CANDIDATES),
            Prefetch(query=sparse_vec, using=SPARSE_VECTOR_NAME, filter=tenant_filter, limit=CANDIDATES),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=CANDIDATES,
        with_payload=True,
    )

    candidates = results.points
    if not candidates:
        print("[RETRIEVE] No candidates returned")
        return []

    print(f"[RETRIEVE] {len(candidates)} candidates from hybrid search")

    # Rerank with cross-encoder
    passages = [
        {"id": i, "text": c.payload.get("text", "")}
        for i, c in enumerate(candidates)
    ]
    reranked = get_ranker().rerank(RerankRequest(query=query, passages=passages))

    # Convert to LangChain Documents
    documents = []
    for result in reranked[:k]:
        payload = candidates[result["id"]].payload
        documents.append(Document(
            page_content=payload["text"],
            metadata={
                "chunk_id": payload.get("chunk_id"),
                "document_id": payload.get("document_id"),
                "section_title": payload.get("section_title"),
                "section_path": payload.get("section_path"),
                "page_hint": payload.get("page_hint"),
                "filename": payload.get("filename"),
                "document_title": payload.get("document_title"),
                "is_table": payload.get("is_table", False),
                "element_type": payload.get("element_type"),
                "upload_date": payload.get("upload_date"),
                "rerank_score": float(result.get("score", 0)),
            }
        ))

    print(f"[RETRIEVE] Returning {len(documents)} reranked documents")
    return documents