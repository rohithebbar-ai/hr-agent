"""
Redis Client
─────────────
Shared Redis connection for caching, rate limiting, and future job queues.

Usage:
    from api.redis_client import get_redis
    r = get_redis()
    r.set("key", "value")
    r.get("key")
"""

import os
import redis
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

_redis_client: redis.Redis | None = None

def get_redis() -> Optional[redis.Redis]:
    """
    Get the redis client
    Creates connection on first call, reuses afterward
    """
    global _redis_client

    if _redis_client is None:
        _redis_client = redis.from_url(
            REDIS_URL,
            decode_responses = True, # return strings, not bytes
            socket_connect_timeout = 5,
            retry_on_timeout = True,
        )
        # verify the connection
        try:
            _redis_client.ping()
            print(f"[REDIS] connected to {REDIS_URL}")
        except redis.ConnectionError as e:
            print(f"[REDIS] Connection failed: {e}")
            print("[REDIS] falling back to no caching")
            _redis_client = None
    return _redis_client


def is_redis_available() -> bool:
    """Check if redis is connected and responding"""
    try:
        client = get_redis()
        if client is None:
            return False
        client.ping()
        return True
    except (redis.ConnectionError, redis.TimeoutError):
        return False


def main():
    s = is_redis_available()
    
if __name__ == "__main__":
    main()