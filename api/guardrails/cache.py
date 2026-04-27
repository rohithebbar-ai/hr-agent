"""
Response Cache (Redis-backed)
─────────────────────────────
Caches LLM responses to avoid repeat API calls.
HR policies don't change frequently, so 24-hour TTL is safe.

Expected impact: 30-40% reduction in LLM calls on repeat queries.

Falls back gracefully if Redis is unavailable — the app works
without caching, just slower.
"""
import hashlib
import json

from api.redis_client import get_redis, is_redis_available

# Cache key prefix (namespace to avoid collisions)
CACHE_PREFIX = "vanacihr:cache:"

# DEFAULT TTL: 24 hours
DEFAULT_TTL = 86400

def _make_key(question: str) -> str:
    """
    Normalize a question into cache key
    Lowercased, stripped, extra whitespace collapsed, then hashed.
    """
    normalized = question.lower().strip()
    normalized = " ".join(normalized.split())
    question_hash = hashlib.md5(normalized.encode()).hexdigest()
    return f"{CACHE_PREFIX}{question_hash}"

def cache_get(question: str) -> dict | None:
    """
    Look up a cached response.

    Returns:
        {"answer": "...", "sources": [...]} on hit, None on miss.
    """
    if not is_redis_available():
        return None

    try:
        r = get_redis()
        key = _make_key(question)
        data = r.get(key)

        if data is None:
            r.incr("vanacihr:stats:misses")
            return None
        r.incr("vanacihr:stats:hits")
        return json.loads(data)
    except Exception as e:
        print(f"[CACHE] get error {e}")
        return None

def cache_set(
    question:str,
    answer: str,
    sources: list[str] = None,
    ttl: int = DEFAULT_TTL,
):
    """
    Store a response in the cache with ttl
    """
    try:
        r = get_redis()
        key = _make_key(question)

        data = json.dumps({
            "answer": answer,
            "sources": sources or [],
        })

        r.set(key, data, ex=ttl)
    except Exception as e:
        print(f"[CACHE] set error {e}")


def cache_stats() -> dict:
    """ Return cache hit/miss statistics """
    if not is_redis_available():
        return {
            "status": "redis_unavailable",
            "hits": 0,
            "misses": 0,
            "hit_rate": "0%",
            "cached_responses": 0,
        }

    try:
        r = get_redis()
        hits = int(r.get("vanacihr:stats:hits") or 0)
        misses = int(r.get("vanacihr:stats:misses") or 0)
        total = hits + misses
        hit_rate = f"{hits/ total:.1%}" if total > 0 else "0%"

        # Count cached responses
        keys = r.keys(f"{CACHE_PREFIX}*")
        cached_count = len(keys)

        return {
            "status": "connected",
            "hits": hits,
            "misses": misses,
            "hit_rate": hit_rate,
            "cached_responses": cached_count,
            "ttl_seconds": DEFAULT_TTL,
        }
    except Exception as e:
        return {"status": f"error: {e}"}
        
def cache_clear():
    """Clear all cached responses and reset stats."""
    if not is_redis_available():
        return

    try:
        r = get_redis()
        # Delete all cache keys
        keys = r.keys(f"{CACHE_PREFIX}*")
        if keys:
            r.delete(*keys)

        # Reset stats
        r.set("vanaciretain:stats:hits", 0)
        r.set("vanaciretain:stats:misses", 0)

        print(f"[CACHE] Cleared {len(keys)} entries")

    except Exception as e:
        print(f"[CACHE] Clear error: {e}")


