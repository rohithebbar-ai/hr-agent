"""
Reindex All Documents
──────────────────────
Pulls all complete documents from Neon and pushes each
to the Redis ingestion queue for reprocessing.
 
Uses incremental indexing — only changed chunks are re-embedded.
Suitable for weekly cron runs to pick up pipeline improvements.
 
Usage:
    uv run python -m rag.reindex_all
    uv run python -m rag.reindex_all --tenant_id vanaciprime
    uv run python -m rag.reindex_all --dry_run
"""
 
import argparse
import json
import os
 
import redis
 
from rag.db import SessionLocal, Document
 
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
QUEUE_KEY = "ingestion_jobs"
 
 
def reindex_all(tenant_id: str = None, dry_run: bool = False):
    """
    Queue all complete documents for reprocessing.
    Incremental indexing ensures only changed chunks are re-embedded.
    """
    db = SessionLocal()
    try:
        query = db.query(Document).filter(Document.status == "complete")
        if tenant_id:
            query = query.filter(Document.tenant_id == tenant_id)
        documents = query.all()
    finally:
        db.close()
 
    print(f"[REINDEX] Found {len(documents)} documents to reindex")
    if dry_run:
        print("[REINDEX] Dry run — no jobs queued")
        for doc in documents:
            print(f"  Would reindex: {doc.filename} ({doc.document_id})")
        return
 
    r = redis.from_url(REDIS_URL)
    queued = 0
 
    for doc in documents:
        # Check that the file still exists on disk
        # Worker needs the file to re-process it
        possible_paths = list(
            Path("data/hr_documents/uploads").glob(f"*_{doc.filename}")
        )
 
        if not possible_paths:
            print(f"[REINDEX] File not found on disk — skipping: {doc.filename}")
            continue
 
        file_path = str(possible_paths[0])
        job = {
            "job_id": f"reindex_{doc.document_id[:8]}",
            "file_path": file_path,
            "filename": doc.filename,
            "document_id": doc.document_id,
            "tenant_id": doc.tenant_id,
            "document_type": doc.document_type or "handbook",
            "use_llamaparse": True,
        }
        r.rpush(QUEUE_KEY, json.dumps(job))
        queued += 1
        print(f"[REINDEX] Queued: {doc.filename}")
 
    print(f"[REINDEX] Done — {queued}/{len(documents)} documents queued")
 
 
if __name__ == "__main__":
    from pathlib import Path
 
    parser = argparse.ArgumentParser(description="Reindex all documents")
    parser.add_argument(
        "--tenant_id",
        help="Only reindex documents for a specific tenant",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be queued without actually queuing",
    )
    args = parser.parse_args()
    reindex_all(tenant_id=args.tenant_id, dry_run=args.dry_run)