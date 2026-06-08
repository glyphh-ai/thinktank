"""Tests for the in-memory cache (dev backend) and selection."""

from __future__ import annotations

import time

from app.cache import InMemoryCache, get_cache
from app.config import Settings


def test_set_get_delete():
    c = InMemoryCache()
    c.set("k", "v")
    assert c.get("k") == "v"
    c.delete("k")
    assert c.get("k") is None


def test_ttl_expiry():
    c = InMemoryCache()
    c.set("k", "v", ttl_seconds=1)
    assert c.get("k") == "v"
    time.sleep(1.1)
    assert c.get("k") is None


def test_incr_counter():
    c = InMemoryCache()
    assert c.incr("hits", ttl_seconds=60) == 1
    assert c.incr("hits") == 2
    assert c.incr("hits") == 3


def test_get_cache_defaults_to_memory():
    assert isinstance(get_cache(Settings(cache_backend="memory")), InMemoryCache)
