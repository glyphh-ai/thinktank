"""step1 — build contrast axes from v07 vectors and project the targets.

Reuses v07's extraction verbatim (`vectors.npz`, layer picked by split-half
reliability, centered, unit-norm). An axis is the unit-normalized difference
v(A) - v(B) of two prime vectors. Each of the 100 targets gets a 13-dim
projection profile.

No new forward passes; run v07's step1_vectors.py first if vectors.npz is
missing.
"""
import os
import sys

import numpy as np

V07 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "version_07_prime_basis")
sys.path.insert(0, V07)

from concepts import TARGETS  # noqa: E402
from step2_basis import load_vectors  # noqa: E402

from axes import AXES, AXIS_NAMES  # noqa: E402

VECTORS = os.path.join(V07, "vectors.npz")


def axis_matrix(vecs, pairs):
    """Unit-norm contrast directions for (A, B) word pairs. -> [n_axes, d]"""
    D = np.stack([vecs[a] - vecs[b] for a, b in pairs])
    return D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)


def project(vecs, pairs, words=TARGETS):
    """Projection profiles of `words` onto the pair axes. -> [n_words, n_axes]"""
    A = axis_matrix(vecs, pairs)
    V = np.stack([vecs[w] for w in words])
    return V @ A.T


def load_all(split="both"):
    vecs, layer, rel, _ = load_vectors(VECTORS, split=split)
    return vecs, layer, rel


def main():
    vecs, layer, rel = load_all()
    P = project(vecs, AXES)
    print(f"layer {layer} (reliability {rel:.3f}) — profiles: {P.shape}")
    print(f"{'axis':<20s} {'most A-side':<28s} most B-side")
    for i, name in enumerate(AXIS_NAMES):
        order = np.argsort(-P[:, i])
        hi = ", ".join(TARGETS[j] for j in order[:4])
        lo = ", ".join(TARGETS[j] for j in order[-4:][::-1])
        print(f"{name:<20s} {hi:<28s} {lo}")


if __name__ == "__main__":
    main()
