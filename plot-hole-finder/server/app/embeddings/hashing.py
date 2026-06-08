"""Deterministic hashing embedder -- the offline/dev default.

Feature-hashes word unigrams+bigrams into a fixed-dim vector. Not semantic, but
deterministic, dependency-free, and fast -- so the full pipeline (and CI) runs
with zero model downloads or network. Swap to FastEmbed in prod for real
semantics; the dimension and call site are identical.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from .base import Embedder, l2_normalize

_TOKEN_RE = re.compile(r"\w+")


class HashingEmbedder(Embedder):
    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(text.lower())
            grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for gram in grams:
                h = int.from_bytes(hashlib.md5(gram.encode()).digest()[:8], "little")
                idx = h % self.dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[i, idx] += sign
        return l2_normalize(out)
