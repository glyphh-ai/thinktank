"""The FROZEN v10 pipeline, applied to words it has never seen.

Nothing here is fit on benchmark data. The whole calibration — layer 5, the
1830-axis bank order, per-axis (mu, sigma, tau), and the v09 centering mean —
was frozen in v10 on the v09 web-frequency vocabulary. A new word's barcode:

    v      = raw mean vector at layer 5 (12-frame protocol)
    v'     = (v - g09) / ||v - g09||          g09 = v09 grand mean, frozen
    z_i    = (v' . axis_i - mu_i) / sigma_i    frozen per-axis standardization
    bit_i  = sign(z_i) if |z_i| >= tau_i else 0

This is the production path: extraction plus a lookup table.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V09 = os.path.join(HERE, "..", "version_09_semantic_barcode")
V10 = os.path.join(HERE, "..", "version_10_ternary_barcode")
sys.path.insert(0, V09)
sys.path.insert(0, V10)

from ternary import load_bank  # noqa: E402  (v10)

BENCH_NPZ = os.path.join(HERE, "bench_vectors.npz")
V09_NPZ = os.path.join(V09, "vocab_vectors.npz")


def _pooled(npz, layer):
    z = np.load(npz, allow_pickle=True)
    words = [str(w) for w in z["words"]]
    splits = [str(s) for s in z["split_names"]]
    means = z["sums"] / z["counts"][None, :, None, None]
    ia, ib = splits.index("half_a"), splits.index("half_b")
    return words, (means[:, ia, layer, :] + means[:, ib, layer, :]) / 2


def load_frozen():
    bank = load_bank()
    layer = int(bank["layer"])
    _, V09_raw = _pooled(V09_NPZ, layer)
    g09 = V09_raw.mean(0)
    return bank, layer, g09


def bench_vectors(layer, g09):
    """Benchmark words -> frozen-transformed unit vectors."""
    words, raw = _pooled(BENCH_NPZ, layer)
    V = raw - g09
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return words, V


def zscores(V, bank, K=None):
    K = K or len(bank["axes"])
    return (V @ bank["axes"][:K].T - bank["mu"][:K]) / bank["sd"][:K]


def ternary_codes(V, bank, K=None):
    z = zscores(V, bank, K)
    K = z.shape[1]
    c = np.sign(z) * (np.abs(z) >= bank["tau"][:K])
    return c.astype(np.int8)


def code_cos(Ca, Cb):
    """Row-wise cosine between two code matrices (paired rows)."""
    a, b = Ca.astype(np.float32), Cb.astype(np.float32)
    return (a * b).sum(1) / ((np.linalg.norm(a, axis=1) + 1e-9) * (np.linalg.norm(b, axis=1) + 1e-9))
