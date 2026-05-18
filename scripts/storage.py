"""
Storage Helper
──────────────
Environment-aware file access. Reads from local filesystem in dev,
from S3 in production. Code doesn't need to know which one.

Usage:
    from scripts.storage import get_document_path, download_document

    # Automatically reads from local or S3
    pdf_path = download_document("raw/gallagher_employee_handbook.pdf")
    policies_path = get_document_path("processed/policies.json")
"""

import os
import tempfile
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")
S3_BUCKET = os.environ.get("S3_BUCKET", "vanacihr-documents")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL")  # Set for LocalStack

# Local paths (relative to project root)
LOCAL_RAW_DIR = Path("data/hr_documents/raw")
LOCAL_PROCESSED_DIR = Path("data/hr_documents/processed")


def is_local() -> bool:
    """Check if we're running locally (dev) or on AWS (prod/staging)."""
    return ENVIRONMENT in ("dev", "test", "local")


def _get_s3_client():
    """Get S3 client — auto-detects LocalStack vs real AWS."""
    kwargs = {
        "service_name": "s3",
        "region_name": AWS_REGION,
    }
    if AWS_ENDPOINT_URL:
        # LocalStack
        kwargs["endpoint_url"] = AWS_ENDPOINT_URL
        kwargs["aws_access_key_id"] = "test"
        kwargs["aws_secret_access_key"] = "test"
    # On EC2, boto3 auto-discovers credentials from IAM role
    return boto3.client(**kwargs)


def get_document_path(key: str) -> Path:
    """
    Get the local path to a document.
    In dev: returns the local filesystem path directly.
    In prod: downloads from S3 to a temp file and returns that path.

    Args:
        key: The document key (e.g., "raw/gallagher_employee_handbook.pdf"
             or "processed/policies.json")

    Returns:
        Path to the file on local disk
    """
    if is_local():
        # Dev: just return the local path
        local_path = Path("data/hr_documents") / key
        if local_path.exists():
            return local_path
        raise FileNotFoundError(
            f"Local file not found: {local_path}. "
            f"Make sure the file exists in data/hr_documents/"
        )

    # Prod: download from S3
    return download_from_s3(key)


def download_from_s3(key: str) -> Path:
    """Download a file from S3 to a temp location."""
    s3 = _get_s3_client()
    suffix = Path(key).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

    try:
        s3.download_file(S3_BUCKET, key, tmp.name)
        print(f"[S3 DOWNLOAD] {key} -> {tmp.name}")
        return Path(tmp.name)
    except Exception as e:
        os.unlink(tmp.name)
        raise RuntimeError(f"Failed to download {key} from S3: {e}")


def upload_to_s3(local_path: Path, key: str):
    """Upload a local file to S3."""
    if is_local() and not AWS_ENDPOINT_URL:
        print(f"[STORAGE] Skipping S3 upload in dev: {key}")
        return

    s3 = _get_s3_client()
    s3.upload_file(str(local_path), S3_BUCKET, key)
    print(f"[S3 UPLOAD] {local_path} -> s3://{S3_BUCKET}/{key}")


def save_document(data: str | bytes, key: str, local_dir: Path = None):
    """
    Save a document both locally and to S3.
    In dev: saves only locally.
    In prod: saves locally AND uploads to S3.
    """
    # Always save locally
    if local_dir is None:
        local_dir = LOCAL_PROCESSED_DIR
    local_path = local_dir / Path(key).name
    local_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if isinstance(data, str) else "wb"
    with open(local_path, mode) as f:
        f.write(data)
    print(f"[STORAGE] Saved locally: {local_path}")

    # In prod, also upload to S3
    if not is_local():
        upload_to_s3(local_path, key)

    return local_path


def list_documents(prefix: str = "") -> list[str]:
    """List available documents."""
    if is_local():
        base = Path("data/hr_documents")
        files = []
        for f in base.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(base))
                if rel.startswith(prefix):
                    files.append(rel)
        return files

    s3 = _get_s3_client()
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", [])]