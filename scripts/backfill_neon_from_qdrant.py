"""
Backfill Neon documents table from existing Qdrant chunks.

Reads chunk payloads from Qdrant Cloud, groups by document_id,
and inserts a Document row in Neon for each unique document.

Usage:
    uv run python scripts/backfill_neon_from_qdrant.py
"""

import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from rag.db import SessionLocal, Document
from rag.pipeline.upserter import COLLECTION_NAME


def backfill():
    # ── Connect to Qdrant Cloud ──────────────────────────────────────────────
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")
    client = QdrantClient(url=url, api_key=api_key, timeout=60) if api_key else QdrantClient(url=url, timeout=60)

    print(f"[BACKFILL] Scrolling chunks from Qdrant: {url}")

    # ── Scroll all points ────────────────────────────────────────────────────
    docs: dict[str, dict] = {}   # document_id → metadata
    offset = None

    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            p = point.payload
            doc_id = p.get("document_id")
            if not doc_id or doc_id in docs:
                continue
            docs[doc_id] = {
                "document_id": doc_id,
                "filename":      p.get("filename", "unknown"),
                "file_type":     p.get("file_type", ""),
                "document_type": p.get("document_type", "Employee Handbook"),
                "tenant_id":     p.get("tenant_id", "vanaciprime"),
            }
        if offset is None:
            break

    print(f"[BACKFILL] Found {len(docs)} unique documents in Qdrant")

    # ── Count chunks per document ────────────────────────────────────────────
    chunk_counts: dict[str, int] = {}
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=["document_id"],
            with_vectors=False,
        )
        for point in results:
            doc_id = point.payload.get("document_id")
            if doc_id:
                chunk_counts[doc_id] = chunk_counts.get(doc_id, 0) + 1
        if offset is None:
            break

    # ── Write to Neon ────────────────────────────────────────────────────────
    db = SessionLocal()
    inserted = 0
    skipped = 0
    try:
        for doc_id, meta in docs.items():
            existing = db.query(Document).filter(Document.document_id == doc_id).first()
            if existing:
                skipped += 1
                continue

            doc = Document(
                document_id=doc_id,
                filename=meta["filename"],
                file_type=meta["file_type"],
                document_type=meta["document_type"],
                tenant_id=meta["tenant_id"],
                chunk_count=chunk_counts.get(doc_id, 0),
                upload_date=datetime.utcnow(),
                status="complete",
                loader_used="backfill",
            )
            db.add(doc)
            inserted += 1
            print(f"  ✓ {meta['filename']}  ({chunk_counts.get(doc_id, 0)} chunks)")

        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[BACKFILL] DB error: {e}")
        raise
    finally:
        db.close()

    print(f"\n[BACKFILL] Done — inserted: {inserted}, skipped (already existed): {skipped}")


if __name__ == "__main__":
    backfill()
