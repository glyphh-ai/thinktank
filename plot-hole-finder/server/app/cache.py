"""Cache abstraction -- in-memory for dev, Redis for prod.

One interface so call sites never branch. Redis is selected via env
(``CACHE_BACKEND=redis`` + ``REDIS_URL``); dev uses a process-local TTL cache so
nothing extra is required to run. The same Redis instance is the natural home for
rate-limiting and (later) a job broker.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from .config import Settings, get_settings


class Cache(ABC):
    @abstractmethod
    def get(self, key: str) -> str | None: ...

    @abstractmethod
    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def incr(self, key: str, ttl_seconds: int | None = None) -> int:
        """Atomically increment a counter (for rate limiting); returns new value."""


class InMemoryCache(Cache):
    """Dev backend: process-local dict with TTLs. Not shared across processes."""

    def __init__(self):
        self._store: dict[str, tuple[str, float | None]] = {}

    def _live(self, key: str) -> str | None:
        item = self._store.get(key)
        if item is None:
            return None
        value, expires = item
        if expires is not None and expires < time.time():
            self._store.pop(key, None)
            return None
        return value

    def get(self, key: str) -> str | None:
        return self._live(key)

    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        expires = time.time() + ttl_seconds if ttl_seconds else None
        self._store[key] = (value, expires)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def incr(self, key: str, ttl_seconds: int | None = None) -> int:
        current = int(self._live(key) or 0) + 1
        existing = self._store.get(key)
        expires = existing[1] if existing else (time.time() + ttl_seconds if ttl_seconds else None)
        self._store[key] = (str(current), expires)
        return current


class RedisCache(Cache):
    """Prod backend: shared Redis (redis-py, lazy import)."""

    def __init__(self, url: str):
        import redis  # lazy

        self._r = redis.Redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> str | None:
        return self._r.get(key)

    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        self._r.set(key, value, ex=ttl_seconds)

    def delete(self, key: str) -> None:
        self._r.delete(key)

    def incr(self, key: str, ttl_seconds: int | None = None) -> int:
        value = self._r.incr(key)
        if value == 1 and ttl_seconds:
            self._r.expire(key, ttl_seconds)
        return int(value)


def get_cache(settings: Settings | None = None) -> Cache:
    settings = settings or get_settings()
    if settings.cache_backend == "redis":
        return RedisCache(settings.redis_url)
    return InMemoryCache()
