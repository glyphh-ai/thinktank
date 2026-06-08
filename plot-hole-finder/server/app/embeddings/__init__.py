"""Passage embeddings (RAG evidence layer) and vector search."""

from ..config import Settings, get_settings
from .base import Embedder
from .hashing import HashingEmbedder
from .search import Neighbor, search_passages


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.embed_backend == "fastembed":
        from .fastembed_embedder import FastEmbedEmbedder

        return FastEmbedEmbedder()
    return HashingEmbedder()


__all__ = ["Embedder", "HashingEmbedder", "get_embedder", "Neighbor", "search_passages"]
