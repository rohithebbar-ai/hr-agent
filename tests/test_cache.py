import pytest
from api.guardrails.cache import cache_clear, cache_get, cache_set, cache_stats
from api.redis_client import is_redis_available

@pytest.fixture(autouse=True)
def clear_before_each():
    if is_redis_available():
        cache_clear()
    yield

@pytest.mark.skipif(not is_redis_available(), reason="redis not running")
class TestCache:
    def test_set_and_get(self):
        cache_set("vacation days?", "15 days", ["Vacation Policy"])
        result = cache_get("vacation days?")
        assert result is not None
        assert result["answer"] == "15 days"

    def test_miss_returns_none(self):
        result = cache_get("never asked this")
        assert result is None

    def test_case_insensitive(self):
        cache_set("Vacation Days?", "15 days")
        result = cache_get("vacation days?")
        assert result is not None

    def test_stats_work(self):
        stats = cache_stats()
        assert "hits" in stats
        assert "misses" in stats