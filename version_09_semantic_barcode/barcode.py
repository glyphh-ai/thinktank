"""Shared core for version_09: loading, axis construction, barcode computation.

Layer choice is a measurement criterion, computed once on the eval-vocabulary
extraction (median split-half cosine after centering), blind to every
downstream test. Prime vectors (for axes) are read from v07's vectors.npz at
that same layer. Contrast axes v(A)-v(B) are centering-invariant, so only the
eval vectors get centered (by the eval grand mean).

A barcode is sign(V @ axes.T): one bit per (word, axis).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V07 = os.path.join(HERE, "..", "version_07_prime_basis")
sys.path.insert(0, V07)

from concepts import CONTROL_POOL, PRIMES  # noqa: E402

VOCAB_NPZ = os.path.join(HERE, "vocab_vectors.npz")
V07_NPZ = os.path.join(V07, "vectors.npz")
AXES_NPZ = os.path.join(HERE, "axes.npz")


def _means(npz):
    z = np.load(npz, allow_pickle=True)
    words = [str(w) for w in z["words"]]
    splits = [str(s) for s in z["split_names"]]
    means = z["sums"] / z["counts"][None, :, None, None]
    return words, splits, means


def best_layer(npz=VOCAB_NPZ):
    """Median split-half cosine per layer (centered); return argmax layer."""
    _, splits, means = _means(npz)
    ia, ib = splits.index("half_a"), splits.index("half_b")
    rels = []
    for layer in range(means.shape[2]):
        a = means[:, ia, layer, :] - means[:, ia, layer, :].mean(0)
        b = means[:, ib, layer, :] - means[:, ib, layer, :].mean(0)
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        rels.append(np.median(cos))
    return int(np.argmax(rels)), float(np.max(rels))


def eval_vectors(layer, npz=VOCAB_NPZ, split="pooled"):
    """Centered (eval grand mean), unit-norm eval-vocab vectors at `layer`."""
    words, splits, means = _means(npz)
    ia, ib = splits.index("half_a"), splits.index("half_b")
    if split == "pooled":
        V = (means[:, ia, layer, :] + means[:, ib, layer, :]) / 2
    else:
        V = means[:, splits.index(split), layer, :]
    V = V - V.mean(0)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return words, V


def word_vectors_v07(layer, which):
    """Raw (uncentered) v07 mean vectors at `layer` for the given word list —
    used only to build difference axes, where centering cancels."""
    words, splits, means = _means(V07_NPZ)
    ia, ib = splits.index("half_a"), splits.index("half_b")
    V = (means[:, ia, layer, :] + means[:, ib, layer, :]) / 2
    lut = {w: V[i] for i, w in enumerate(words)}
    return np.stack([lut[w] for w in which])


def pair_axes(vectors, pairs_idx):
    """Unit-norm difference axes for index pairs into `vectors`."""
    D = np.stack([vectors[i] - vectors[j] for i, j in pairs_idx])
    return D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)


def all_prime_pairs():
    n = len(PRIMES)
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def codes(V, axes):
    """Barcodes: sign of projections. -> int8 [n_words, n_axes] in {-1, +1}."""
    s = np.sign(V @ axes.T)
    s[s == 0] = 1
    return s.astype(np.int8)


def code_sim_upper(C):
    """Normalized code agreement (equivalent to 1 - 2*Hamming) upper triangle."""
    S = (C.astype(np.float32) @ C.T.astype(np.float32)) / C.shape[1]
    return S[np.triu_indices(len(C), k=1)]


def cos_sim_upper(V):
    U = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    S = U @ U.T
    return S[np.triu_indices(len(V), k=1)]


def load_axis_bank():
    z = np.load(AXES_NPZ, allow_pickle=True)
    return z["axes"], [str(n) for n in z["names"]], z["stability"], int(z["layer"])
