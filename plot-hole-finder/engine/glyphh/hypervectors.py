"""Core hyperdimensional vector operations (bipolar VSA / HRR).

Hypervectors are dense bipolar arrays in {-1, +1}^D. The algebra:

- bind    (otimes): elementwise product. Self-inverse (a*a = 1), invertible
                    (a*b*b = a), and similarity-randomizing. Used to attach a
                    value to a role: ``bind(role, value)``.
- bundle  (oplus):  elementwise majority (sign of the sum). Similarity
                    *preserving* -- a bundle stays correlated with each of its
                    inputs. This is what makes "similarity at every level"
                    possible, but it is lossy with finite capacity, so we only
                    *unbind* at the leaf (segment) level.
- permute (rho):    cyclic shift. Similarity-randomizing but order-preserving;
                    used to protect sequence position when needed.

Everything here is pure numpy. Hypervectors are ``int8`` bipolar arrays;
accumulators (running, un-thresholded bundles) are ``int64``.
"""

from __future__ import annotations

import numpy as np

Hypervector = np.ndarray


def random_hv(dim: int, rng: np.random.Generator) -> Hypervector:
    """Draw a fresh random bipolar hypervector in {-1, +1}^dim."""
    return (rng.integers(0, 2, size=dim, dtype=np.int8) * 2 - 1).astype(np.int8)


def bind(*vectors: Hypervector) -> Hypervector:
    """Bind hypervectors with the elementwise product.

    Self-inverse and invertible: ``bind(bind(a, b), b) == a``. Binding a role
    with a value produces a vector dissimilar to both -- the pair is recovered
    later by binding again with the known factor (unbind).
    """
    if not vectors:
        raise ValueError("bind() requires at least one hypervector")
    out = vectors[0].astype(np.int8)
    for v in vectors[1:]:
        out = (out * v).astype(np.int8)
    return out


# Unbinding is identical to binding for bipolar vectors (product is its own
# inverse); the alias documents intent at call sites.
unbind = bind


def bundle(vectors: list[Hypervector], rng: np.random.Generator | None = None) -> Hypervector:
    """Bundle (superpose) hypervectors via elementwise majority.

    The result is bipolar and similar to every input. Ties (an even number of
    inputs summing to zero on a component) are broken randomly when ``rng`` is
    supplied, otherwise toward +1.
    """
    if not vectors:
        raise ValueError("bundle() requires at least one hypervector")
    acc = np.sum(np.stack(vectors).astype(np.int64), axis=0)
    return threshold(acc, rng)


def threshold(acc: np.ndarray, rng: np.random.Generator | None = None) -> Hypervector:
    """Collapse an integer accumulator to a bipolar hypervector via sign."""
    out = np.sign(acc).astype(np.int8)
    ties = out == 0
    if ties.any():
        if rng is not None:
            out[ties] = (rng.integers(0, 2, size=int(ties.sum()), dtype=np.int8) * 2 - 1)
        else:
            out[ties] = 1
    return out


def permute(v: Hypervector, shift: int = 1) -> Hypervector:
    """Cyclically shift a hypervector to encode order/position."""
    return np.roll(v, shift)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Works on bipolar vectors and int accumulators alike."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
