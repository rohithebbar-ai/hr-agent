"""
Rate Limiter
────────────
Shared slowapi limiter instance backed by Redis.
Imported by both main.py and routes.py to avoid circular imports.
"""

import os

from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address

load_dotenv()

import os
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address

load_dotenv()

_env = os.environ.get("ENVIRONMENT", "dev")
_rate_limit = "1000/minute" if _env == "test" else os.environ.get("RATE_LIMIT", "10/minute")
_storage_uri = "memory://" if _env == "test" else os.environ.get("REDIS_URL", "redis://localhost:6379/0")

print(f"[LIMITER] Rate limit: {_rate_limit} (env: {_env})")

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[_rate_limit],
    storage_uri=_storage_uri,
)