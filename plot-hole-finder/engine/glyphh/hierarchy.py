"""The glyphh hierarchy: roles -> segments -> layers -> cortex.

Design rules (see ../DESIGN.md section 5a):

1. Bundle children *raw* so similarity is preserved at every level. Addressing
   is done through each node's cleanup memory, never by id-unbinding.
2. Unbind only at the leaf (segment). Layer and cortex are routing-only: their
   thresholded vectors are for cosine similarity, not value recovery.
3. Each node keeps both a running integer accumulator (pre-threshold, preserves
   multiplicity for probing) and exposes a bipolar ``vector`` (for cross-node
   similarity). Probes read the accumulator; routing reads the vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hypervectors import Hypervector, bind, similarity, threshold
from .memory import CleanupMemory, Codebook, Match


@dataclass
class Role:
    """A single typed assertion: ``key (x) value`` (e.g. EYE_COLOR (x) BROWN)."""

    key: str
    value: str
    vector: Hypervector


def make_role(codebook: Codebook, key: str, value: str) -> Role:
    return Role(key, value, bind(codebook[key], codebook[value]))


class _Node:
    """Shared accumulator/threshold/similarity machinery."""

    def __init__(self, name: str, dim: int):
        self.name = name
        self.dim = dim
        self._acc = np.zeros(dim, dtype=np.int64)

    @property
    def accumulator(self) -> np.ndarray:
        return self._acc

    @property
    def vector(self) -> Hypervector:
        """Bipolar (thresholded) view for cross-node similarity / routing."""
        return threshold(self._acc)

    def similarity(self, other: "_Node | Hypervector") -> float:
        ov = other.vector if isinstance(other, _Node) else other
        return similarity(self.vector, ov)


class Segment(_Node):
    """An entity-state snapshot (or scene): a raw bundle of roles.

    The leaf level -- the only place we unbind to recover values.
    """

    def __init__(self, name: str, dim: int, narrative_time: int = 0):
        super().__init__(name, dim)
        self.narrative_time = narrative_time
        self.roles: list[Role] = []

    def add(self, role: Role) -> "Segment":
        self._acc += role.vector.astype(np.int64)
        self.roles.append(role)
        return self

    def probe(self, codebook: Codebook, key: str, values: CleanupMemory) -> Match | None:
        """Unbind ``key`` from this segment and clean up to the nearest value."""
        recovered = bind(self._acc.astype(np.int64), codebook[key].astype(np.int64))
        return values.best(recovered)


class Layer(_Node):
    """An entity timeline (or chapter): a raw bundle of segments.

    Routing-only at the thresholded level; probing the accumulator surfaces
    *contested* attributes (a role holding several values across the timeline).
    """

    def __init__(self, name: str, dim: int):
        super().__init__(name, dim)
        self.segments: list[Segment] = []
        self.cleanup = CleanupMemory(dim)  # segment vectors, for drill-down

    def add(self, segment: Segment) -> "Layer":
        self._acc += segment.vector.astype(np.int64)
        self.segments.append(segment)
        self.cleanup.add(segment.name, segment.vector)
        return self

    def route(self, query: Hypervector) -> Match | None:
        """Find the segment in this layer most similar to ``query``."""
        return self.cleanup.best(query)

    def contested(
        self, codebook: Codebook, key: str, values: CleanupMemory, threshold_sim: float = 0.06
    ) -> list[Match]:
        """Values of ``key`` present across the timeline above the noise floor.

        More than one => candidate inconsistency to localize at the segments.
        """
        recovered = bind(self._acc.astype(np.int64), codebook[key].astype(np.int64))
        return values.above(recovered, threshold_sim)

    def localize(
        self, codebook: Codebook, key: str, values: CleanupMemory
    ) -> dict[str, list[Segment]]:
        """Group this layer's segments by their cleaned-up value for ``key``."""
        groups: dict[str, list[Segment]] = {}
        for seg in self.segments:
            m = seg.probe(codebook, key, values)
            if m is not None:
                groups.setdefault(m.label, []).append(seg)
        return groups


class Cortex(_Node):
    """The whole work: a raw bundle of layers. Coarse router across entities."""

    def __init__(self, name: str, dim: int):
        super().__init__(name, dim)
        self.layers: dict[str, Layer] = {}
        self.cleanup = CleanupMemory(dim)  # layer vectors, for routing

    def add(self, layer: Layer) -> "Cortex":
        self._acc += layer.vector.astype(np.int64)
        self.layers[layer.name] = layer
        self.cleanup.add(layer.name, layer.vector)
        return self

    def route(self, query: Hypervector) -> Match | None:
        """Route a stray mention to its most likely entity layer (coreference)."""
        return self.cleanup.best(query)
