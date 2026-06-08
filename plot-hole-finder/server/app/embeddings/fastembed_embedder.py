"""FastEmbed-backed embedder -- the small production embedding model.

Uses BAAI/bge-small-en-v1.5 (384-dim, ONNX, CPU-friendly, no torch). The model
is downloaded lazily on first use, so this is only imported when selected.
"""

from __future__ import annotations

import numpy as np

from .base import Embedder, l2_normalize

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dims, matches EMBEDDING_DIM


class FastEmbedEmbedder(Embedder):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from fastembed import TextEmbedding  # lazy import

        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self._model.embed(texts)), dtype=np.float32)
        return l2_normalize(vectors)
