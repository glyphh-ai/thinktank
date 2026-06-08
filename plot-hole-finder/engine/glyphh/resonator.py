"""Resonator network: factor a product hypervector into its codebook factors.

Given ``p = bind(f1, f2, ..., fn)`` where each ``fi`` is drawn from a known
codebook, recover every factor at once. Plain unbinding only works when all but
one factor is known; the resonator solves the multi-unknown case by iterating
estimates that "resonate" against their codebooks until they lock onto clean
items (Frady, Kent, Olshausen & Sommer, 2020).

Used where a fact is encoded as a single product of several unknown roles and we
need to decompose it cleanly.
"""

from __future__ import annotations

import numpy as np

from .hypervectors import Hypervector, threshold


def resonate(
    product: Hypervector,
    codebooks: list[list[Hypervector]],
    labels: list[list[str]],
    max_iters: int = 100,
    seed: int = 0,
) -> tuple[list[str], int]:
    """Factor ``product`` into one item per codebook.

    Returns the recovered labels and the iteration count at convergence.
    """
    rng = np.random.default_rng(seed)
    dim = product.shape[0]
    mats = [np.stack(cb).astype(np.float64) for cb in codebooks]  # each (k_i, dim)

    # Start each factor estimate as the superposition of its whole codebook.
    estimates = [threshold(m.sum(axis=0)).astype(np.int64) for m in mats]

    prev: list[str] | None = None
    for it in range(1, max_iters + 1):
        new_estimates = []
        for i, mat in enumerate(mats):
            # Unbind the product by every *other* current estimate.
            others = product.astype(np.int64)
            for j, est in enumerate(estimates):
                if j != i:
                    others = others * est
            # Project onto codebook i and re-threshold (the "resonance" step).
            scores = mat @ others.astype(np.float64)
            cleaned = mat[int(np.argmax(np.abs(scores)))] * np.sign(scores[np.argmax(np.abs(scores))])
            new_estimates.append(threshold(cleaned.astype(np.int64), rng).astype(np.int64))
        estimates = new_estimates

        decoded = [labels[i][int(np.argmax(mat @ estimates[i].astype(np.float64)))] for i, mat in enumerate(mats)]
        if decoded == prev:
            return decoded, it
        prev = decoded

    return prev or [], max_iters
