"""TTLDedupeCache failure path (no network)."""
import asyncio

import pytest
from fastapi import HTTPException

from app.cache import TTLDedupeCache


def test_failure_is_reraised_unchanged_and_not_cached():
    cache = TTLDedupeCache(ttl_seconds=60)
    calls = {"n": 0}

    async def failing():
        calls["n"] += 1
        raise HTTPException(status_code=502, detail="upstream down")

    async def ok():
        calls["n"] += 1
        return 42

    async def scenario():
        with pytest.raises(HTTPException) as exc:
            await cache.get_or_compute("k", failing)
        assert exc.value.status_code == 502
        assert await cache.get_or_compute("k", ok) == 42      # retried, not a cached failure
        assert await cache.get_or_compute("k", failing) == 42  # success is cached
        return calls["n"]

    assert asyncio.run(scenario()) == 2
