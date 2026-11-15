"""A tiny in-memory, TTL'd, request-coalescing cache.

This is the *only* caching the backend does — no disk, no database, no
stored images (see PLAN.md / the brief's "stateless processing pipe"
requirement). Its only job is to stop two users (or two tabs) asking for the
exact same thing seconds apart from making IRSA do the work twice: truly
concurrent identical requests share one in-flight `asyncio.Task`, and a
result already computed in the last `ttl_seconds` is returned straight away.
"""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class TTLDedupeCache:
    def __init__(self, ttl_seconds: float = 180.0):
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()
        self._inflight: dict[str, asyncio.Task] = {}
        self._done: dict[str, tuple[float, object]] = {}

    async def get_or_compute(self, key: str, compute: Callable[[], Awaitable[T]]) -> T:
        now = time.time()
        async with self._lock:
            cached = self._done.get(key)
            if cached is not None and now - cached[0] < self._ttl:
                return cached[1]
            task = self._inflight.get(key)
            if task is None:
                task = asyncio.ensure_future(compute())
                self._inflight[key] = task

        try:
            result = await task
        except BaseException:
            # A failure is never cached (the next request retries), and the
            # original exception (e.g. an HTTPException 502) is re-raised as
            # is. Previously this path read an unset `result` and turned every
            # upstream failure into an opaque 500.
            async with self._lock:
                self._inflight.pop(key, None)
            raise
        async with self._lock:
            self._inflight.pop(key, None)
            self._done[key] = (time.time(), result)
            self._sweep_locked()
        return result

    def _sweep_locked(self) -> None:
        """Caller must hold self._lock."""
        now = time.time()
        expired = [k for k, (ts, _) in self._done.items() if now - ts >= self._ttl]
        for k in expired:
            del self._done[k]

    def size(self) -> int:
        return len(self._done) + len(self._inflight)
