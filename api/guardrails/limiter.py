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

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Use Redis as the storage backend so rate limit counters
# persist across server restarts and work with multiple workers.
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=REDIS_URL,
    default_limits=["60/minute"],  # Global default
)