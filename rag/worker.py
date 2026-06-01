# rag/worker.py

import json
import os
import time
from pathlib import Path
import redis

from rag.pipeline import ingest_document
from rag.db import SessionLocal, Document

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
QUEUE_KEY = "ingestion_jobs"

def get_redis_client():
    return redis.from_url(REDIS_URL)

def _update_job_status(r, job_id: str, status: str, result: dict = None):
    """Store job status in Redis ffor API polling"""
    data = {"status": status, "job_id": job_id}
    if result:
        data["result"] = result
    r.setex(f"job:{job_id}", 86400, json.dumps(data))

def _ensure_document_record(job:dict):
    """
    Create a neon document record if it doesn't exist
    Handles case where worker starts before API created the record
    """
    db = SessionLocal()
    try:
        existing = db.query(Document).filter(
            Document.document_id == job.get("document_id", "")
        ).first()

        if not existing and job.get("document_id"):
            doc = Document(
                document_id = job["document_id"],
                filename = job["filename"],
                file_type = Path(job["filename"]).suffix.lstrip("."),
                document_type = job.get("document_type", "handbook"),
                tenant_id = job.get("tenant_id", "vanaciprime"),
                status = "pending",
            )
            db.add(doc)
            db.commit()
    except Exception as e:
        db.rollback()
        print(f"[WORKER] Document record creation failed: {e}")
    finally:
        db.close()

def run():
    print("[WORKER] starting ingestion worker")
    print(f"[WORKER] watching redis queue: {QUEUE_KEY}")

    r = get_redis_client()

    while True:
        try:
            job_data = r.blpop(QUEUE_KEY, timeout=10)

            if job_data is None:
                continue

            _, raw_job = job_data
            job = json.loads(raw_job)
            job_id = job.get("job_id", "unknown")

            print(f"[WORKER] processing job {job_id}: {job['filename']}")
            _update_job_status(r, job_id, "processing")
            _ensure_document_record(job)

            try:
                result = ingest_document(
                    file_path = Path(job['file_path']),
                    filename = job['filename'],
                    tenant_id = job.get("tenant_id", "vanaciprime"),
                    document_type = job.get("document_type", "hanbook"),
                    use_llamaparse = job.get("use_llamaparse", True),
                )
                _update_job_status(r, job_id, "complete", result)
                print(f"[WORKER] Done - {result['chunk_total']} chunks")

            except Exception as e:
                print(f"[WORKER] Failed: {type(e).__name__}: {e}")
                _update_job_status(r, job_id, "failed", {"error": str(e)})
        except redis.exceptions.ConnectionError:
            print("[WORKER] Redis disconnected. Retrying in 5 seconds")
            time.sleep(5)
        except KeyboardInterrupt:
            print("[WORKER] shutting down")
            break

if __name__ == "__main__":
    run()

    