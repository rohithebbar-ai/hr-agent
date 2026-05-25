# rag/pipeline/upserter.py

import os
from typing import List, Set
from qdrant_client import QdrantClient
from qdrant_client.models import(
    PointStruct, VectorParams, Distance,
    SparseVectorParams, SparseIndexParams,
    Filter, FieldCondition, MatchValue,
    SparseVector,
)

COLLECTION_NAME = "hr_documents"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
VECTOR_SIZE = 384

_client: QdrantClient = None

def get_client() -> QdrantClient:
    global _client
    if _client is None:
        url = os.environ.get("QDRANT_URL")
        api_key = os.environ.get("QDRANT_API_KEY")
        _client = QdrantClient(url=url, api_key=api_key, timeout=60) if api_key else QdrantClient(url=url, timeout=60)
    return _client

_collection_ready = False

def ensure_collection_exists():
    global _collection_ready
    if _collection_ready:
        return
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        print(f"[UPSERTER] Created hybrid collection: {COLLECTION_NAME}")

    # Always ensure payload indexes exist (idempotent)
    from qdrant_client.models import PayloadSchemaType
    for field in ["document_id", "tenant_id"]:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # Already exists
    _collection_ready = True

def get_existing_chunk_ids(document_id: str) -> Set[str]:
    """Fetch all chunk_ids currently stored for this document"""
    ensure_collection_exists() 
    client = get_client()
    existing_ids = set()
    offset = None

    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            ]),
            limit=1000,
            offset=offset,
            with_payload=["chunk_id"],
            with_vectors=False,
        )
        for point in results:
            chunk_id = point.payload.get("chunk_id")
            if chunk_id:
                existing_ids.add(chunk_id)
        if offset is None:
            break

    return existing_ids

def delete_chunks_by_ids(chunk_ids: Set[str]):
    """Delete specific chunks by their chunk_id (not document_id)"""
    if not chunk_ids:
        return
    client = get_client()
    # convert chunk_ids to Qdrant point IDs (same conversion as upserting)
    point_ids = [int (cid[:8], 16) for cid in chunk_ids]
    client.delete(
        collection_name = COLLECTION_NAME,
        points_selector = point_ids,
    )
    print(f"[UPSERTER] Deleted {len(chunk_ids)} removed chunks")

def upsert_chunks_incremental(chunks: List, document_id: str, existing_ids: Set[str] = None) -> dict:
    """
    Incremental upsert.
    Pass existing_ids from pipeline to avoid a redundant Qdrant fetch.
    Returns stats: {added, deleted, unchanged, total}
    """
    client = get_client()
    ensure_collection_exists()

    if existing_ids is None:
        existing_ids = get_existing_chunk_ids(document_id)
    new_ids = {chunk.chunk_id for chunk in chunks}

    to_add = new_ids - existing_ids        # New chunks to embed and upsert
    to_delete = existing_ids - new_ids    # Old chunks to remove
    unchanged = new_ids & existing_ids    # Skip entirely

    print(f"[UPSERTER] Incremental diff — add: {len(to_add)}, delete: {len(to_delete)}, skip: {len(unchanged)}")

    # Delete removed chunks
    delete_chunks_by_ids(to_delete)

    # Upsert only new chunks
    new_chunks = [c for c in chunks if c.chunk_id in to_add]
    if not new_chunks:
        print("[UPSERTER] No new chunks to upsert")
        return {"added": 0, "deleted": len(to_delete), "unchanged": len(unchanged), "total": len(chunks)}

    points = []
    for chunk in new_chunks:
        if not chunk.embedding or not chunk.sparse_embedding:
            continue

        point_id = int(chunk.chunk_id[:8], 16)
        points.append(PointStruct(
            id=point_id,
            vector={
                DENSE_VECTOR_NAME: chunk.embedding,
                SPARSE_VECTOR_NAME: SparseVector(
                    indices=chunk.sparse_embedding.indices,
                    values=chunk.sparse_embedding.values,
                ),
            },
            payload={
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "element_type": chunk.element_type,
                "section_title": chunk.section_title,
                "section_path": chunk.section_path,
                "page_hint": chunk.page_hint,
                "is_table": chunk.is_table,
                "token_count": chunk.token_count,
                "filename": chunk.filename,
                "file_type": chunk.file_type,
                "document_title": chunk.document_title,
                "upload_date": chunk.upload_date,
                "tenant_id": chunk.tenant_id,
                "document_type": chunk.document_type,
            },
        ))

    total_added = 0
    for i in range(0, len(points), 50):
        batch = points[i:i + 100]
        client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=False)
        total_added += len(batch)
        print(f"[UPSERTER] {total_added}/{len(points)} new chunks upserted")

    return {
        "added": total_added,
        "deleted": len(to_delete),
        "unchanged": len(unchanged),
        "total": len(chunks),
    }
