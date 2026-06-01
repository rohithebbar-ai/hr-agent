# rag/pipeline/pipeline.py

from pathlib import Path
from datetime import datetime
from typing import Optional

from .loader import load_document
from .chunker import chunk_document
from .embedder import embed_chunks
from .upserter import upsert_chunks_incremental, get_existing_chunk_ids
from rag.db import SessionLocal, Document


def _update_status(document_id: str, status: str, chunk_count: int = None, error: str = None, loader_used: str = None):
    """Update document status using SQLAlchemy session."""
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.document_id == document_id).first()
        if not doc:
            print(f"[PIPELINE] Document not found in Neon: {document_id}")
            return
        doc.status = status
        if chunk_count is not None:
            doc.chunk_count = chunk_count
        if error:
            doc.error_message = error
        if loader_used:
            doc.loader_used = loader_used
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[PIPELINE] Status update failed: {e}")
    finally:
        db.close()


def ingest_document(
    file_path: Path,
    filename: str,
    tenant_id: str = "vanaciprime",
    document_type: str = "handbook",
    upload_date: Optional[str] = None,
    use_llamaparse: bool = True,
) -> dict:
    """
    Full pipeline: any file → Qdrant Cloud + Neon tracking.
    Uses incremental indexing — only embeds changed chunks.
    """
    if upload_date is None:
        upload_date = datetime.utcnow().strftime("%Y-%m-%d")

    print(f"\n{'='*60}\n  Ingesting: {filename}\n{'='*60}\n")

    # Stage 1: Load
    print("[STAGE 1] Loading document")
    loaded = load_document(file_path, filename, use_llamaparse=use_llamaparse)

    # Stage 1b: clean noise
    from .loader import clean_elements 
    loaded.elements = clean_elements(loaded.elements)

    # Update Neon: processing
    _update_status(loaded.document_id, "processing", loader_used=loaded.loader_used)

    try:
        # Stage 2: Chunk
        print("[STAGE 2] Element-aware chunking")
        chunks = chunk_document(loaded)
        print(f"[STAGE 2] {len(chunks)} chunks created")

        # Stage 3: Enrich metadata
        print("[STAGE 3] Enriching metadata")
        for chunk in chunks:
            chunk.filename = loaded.filename
            chunk.file_type = loaded.file_type
            chunk.document_title = loaded.title
            chunk.upload_date = upload_date
            chunk.tenant_id = tenant_id
            chunk.document_type = document_type

        # Stage 4: Incremental embedding — only new chunks
        print("[STAGE 4] Checking existing chunks")
        existing_ids = get_existing_chunk_ids(loaded.document_id)
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if new_chunks:
            print(f"[STAGE 4] Embedding {len(new_chunks)} new chunks "
                  f"(skipping {len(existing_ids)} unchanged)")
            embed_chunks(new_chunks)
            # Merge embeddings back into full chunk list
            embedded_map = {c.chunk_id: c for c in new_chunks}
            for chunk in chunks:
                if chunk.chunk_id in embedded_map:
                    chunk.embedding = embedded_map[chunk.chunk_id].embedding
                    chunk.sparse_embedding = embedded_map[chunk.chunk_id].sparse_embedding
        else:
            print("[STAGE 4] All chunks unchanged — skipping embedding")

        # Stage 5: Incremental upsert to Qdrant Cloud
        print("[STAGE 5] Upserting to Qdrant Cloud")
        stats = upsert_chunks_incremental(chunks, loaded.document_id, existing_ids=existing_ids)

        # Update Neon: complete
        _update_status(
            loaded.document_id,
            "complete",
            chunk_count=stats["total"],
            loader_used=loaded.loader_used,
        )

        result = {
            "document_id": loaded.document_id,
            "filename": filename,
            "pages": loaded.page_count,
            "loader_used": loaded.loader_used,
            "chunks_total": stats["total"],
            "chunks_added": stats["added"],
            "chunks_deleted": stats["deleted"],
            "chunks_unchanged": stats["unchanged"],
            "status": "complete",
        }
        print(f"\n[DONE] {result}")
        return result

    except Exception as e:
        _update_status(loaded.document_id, "failed", error=str(e))
        raise
    