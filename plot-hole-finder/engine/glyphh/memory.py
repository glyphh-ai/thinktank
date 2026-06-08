"""Symbol codebooks and cleanup (item) memory.

A :class:`Codebook` maps human-readable symbols (``"EYE_COLOR"``, ``"BROWN"``)
to fixed random hypervectors, generated lazily and reproducibly from a seed.

A :class:`CleanupMemory` stores labelled hypervectors and resolves a noisy query
back to the nearest stored item -- the indispensable counterpart to bundling,
since superposition is lossy and unbinding returns "value + noise".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hypervectors import Hypervector, random_hv, similarity


class Codebook:
    """Reproducible symbol -> hypervector map with lazy allocation."""

    def __init__(self, dim: int, seed: int = 0):
        self.dim = dim
        self._rng = np.random.default_rng(seed)
        self._symbols: dict[str, Hypervector] = {}

    def get(self, symbol: str) -> Hypervector:
        if symbol not in self._symbols:
            self._symbols[symbol] = random_hv(self.dim, self._rng)
        return self._symbols[symbol]

    __getitem__ = get

    def symbols(self) -> list[str]:
        return list(self._symbols)


@dataclass
class Match:
    label: str
    similarity: float
    margin: float  # gap to the runner-up; high margin == confident, clean read


class CleanupMemory:
    """Nearest-neighbour item memory over labelled hypervectors."""

    def __init__(self, dim: int):
        self.dim = dim
        self._labels: list[str] = []
        self._matrix: np.ndarray | None = None  # (n, dim) float

    def add(self, label: str, vector: Hypervector) -> None:
        row = vector.astype(np.float64)[None, :]
        self._labels.append(label)
        self._matrix = row if self._matrix is None else np.vstack([self._matrix, row])

    def __len__(self) -> int:
        return len(self._labels)

    def cleanup(self, query: np.ndarray, top_k: int = 1) -> list[Match]:
        """Return the ``top_k`` nearest stored items, most similar first."""
        if self._matrix is None:
            return []
        q = query.astype(np.float64)
        qn = np.linalg.norm(q)
        if qn == 0.0:
            return []
        sims = (self._matrix @ q) / (np.linalg.norm(self._matrix, axis=1) * qn)
        order = np.argsort(sims)[::-1]
        out: list[Match] = []
        for rank, idx in enumerate(order[:top_k]):
            runner_up = sims[order[rank + 1]] if rank + 1 < len(order) else 0.0
            out.append(Match(self._labels[idx], float(sims[idx]), float(sims[idx] - runner_up)))
        return out

    def best(self, query: np.ndarray) -> Match | None:
        matches = self.cleanup(query, top_k=2)
        return matches[0] if matches else None

    def above(self, query: np.ndarray, threshold: float) -> list[Match]:
        """All stored items whose similarity to the query exceeds ``threshold``.

        Used for *contested-attribute* triage: probing a bundle for one role and
        finding several values above the noise floor means the role holds more
        than one value across the bundle -- a candidate inconsistency.
        """
        return [m for m in self.cleanup(query, top_k=len(self._labels)) if m.similarity >= threshold]


def value_cleanup(codebook: Codebook, values: list[str]) -> CleanupMemory:
    """Build a cleanup memory over a set of candidate value symbols."""
    cm = CleanupMemory(codebook.dim)
    for v in values:
        cm.add(v, codebook[v])
    return cm
