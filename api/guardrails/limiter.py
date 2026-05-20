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

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

_rate_limit = "1000/minute" if os.environ.get("ENVIRONMENT") == "test" else os.environ.get("RATE_LIMIT", "10/minute")
print(f"[LIMITER] Rate limit: {_rate_limit} (env: {os.environ.get('ENVIRONMENT', 'not set')})")

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[_rate_limit],
    storage_uri=REDIS_URL,
)