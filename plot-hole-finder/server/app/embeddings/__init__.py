"""Passage embeddings (RAG evidence layer) and vector search.

FastEmbed (bge-small) is the only embedder -- no non-semantic fallback. Tests
inject a stub for plumbing.
"""

from ..config import Settings, get_settings
from .base import Embedder
from .search import Neighbor, search_passages


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    from .fastembed_embedder import FastEmbedEmbedder

    return FastEmbedEmbedder()


__all__ = ["Embedder", "get_embedder", "Neighbor", "search_passages"]
