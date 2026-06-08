"""glyphh -- hierarchical hyperdimensional memory for plot-hole detection.

Purpose-built for this repo: roles -> segments -> layers -> cortex, giving
similarity at every level for coarse-to-fine triage of narrative
inconsistencies. See ../DESIGN.md section 5a.
"""

from .hierarchy import Cortex, Layer, Role, Segment, make_role
from .hypervectors import bind, bundle, permute, random_hv, similarity, threshold, unbind
from .memory import CleanupMemory, Codebook, Match, value_cleanup
from .resonator import resonate

__all__ = [
    "Cortex",
    "Layer",
    "Role",
    "Segment",
    "make_role",
    "bind",
    "bundle",
    "permute",
    "random_hv",
    "similarity",
    "threshold",
    "unbind",
    "CleanupMemory",
    "Codebook",
    "Match",
    "value_cleanup",
    "resonate",
]
