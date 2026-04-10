"""
HR RAG ingest data layout (contract for deploys)
------------------------------------------------
- Naive baseline reads ``{naive processed dir}/policies.json`` (default:
  ``data/hr_documents/processed/policies.json``).
- Policy-aware reads ``{policy-aware processed dir}/policies.json`` (default:
  ``data/hr_documents/processed/policy_aware/policies.json``) after the
  policy-aware preprocessing step.

Qdrant: ``QDRANT_RECREATE_COLLECTION`` (default ``false``). In production
(``ENV`` or ``APP_ENV`` = ``production``), recreate requires ``I_KNOW_WHAT_IM_DOING=1``.
"""

import os
from pathlib import Path

_DEFAULT_DATA_ROOT = Path("data/hr_documents")


def _data_root() -> Path:
    return Path(os.getenv("HR_RAG_DATA_ROOT", str(_DEFAULT_DATA_ROOT)))


def naive_processed_dir() -> Path:
    return Path(
        os.getenv(
            "HR_RAG_NAIVE_PROCESSED_DIR",
            str(_data_root() / "processed"),
        )
    )


def policy_aware_processed_dir() -> Path:
    return Path(
        os.getenv(
            "HR_RAG_POLICY_AWARE_PROCESSED_DIR",
            str(_data_root() / "processed" / "policy_aware"),
        )
    )


POLICIES_JSON_NAME = "policies.json"

def golden_test_dir() -> Path:
    return Path(os.getenv("HR_RAG_GOLDEN_TEST_DIR", "data/golden_test_set"))

def golden_test_path() -> Path:
    return golden_test_dir() / "golden_test_set.json"

def naive_policies_path() -> Path:
    return naive_processed_dir() / POLICIES_JSON_NAME

def policy_aware_policies_path() -> Path:
    return policy_aware_processed_dir() / POLICIES_JSON_NAME


def assert_policies_json_exists(path: Path) -> None:
    """Fail fast with a clear path for CI/deploy before ingest."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected policies file for this pipeline is missing: {path.resolve()}\n"
            "Check HR_RAG_NAIVE_PROCESSED_DIR, HR_RAG_POLICY_AWARE_PROCESSED_DIR, or HR_RAG_DATA_ROOT."
        )


def resolve_qdrant_force_recreate(*, env_flag: bool, cli_recreate: bool) -> bool:
    """True if either CLI --recreate-collection or QDRANT_RECREATE_COLLECTION=true."""
    return cli_recreate or env_flag


def qdrant_recreate_from_env() -> bool:
    return os.getenv("QDRANT_RECREATE_COLLECTION", "false").lower() == "true"


def ensure_recreate_allowed_if_production(force_recreate: bool) -> None:
    if not force_recreate:
        return
    env_name = (os.getenv("ENV") or os.getenv("APP_ENV") or "").strip().lower()
    if env_name != "production":
        return
    if os.getenv("I_KNOW_WHAT_IM_DOING", "").strip().lower() in ("1", "true", "yes"):
        return
    raise RuntimeError(
        "Refusing to recreate Qdrant collection in production without I_KNOW_WHAT_IM_DOING=1"
    )
