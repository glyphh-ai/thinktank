"""version_12 core: the wide-seed named bank at semantic depth.

LAYER = 12 — pre-registered from v11's exploratory sweep. Disclosure: that
sweep used Warriner data, so gate G4 (also Warriner) is not fully
independent of the layer choice; G1–G3 (SimLex/WS-353) are — the layer-12
full-vector SimLex score is actually slightly LOWER than layer 5's, so the
choice does not favor the similarity gates.

Axes: unit differences of seed-word vectors (242 named pairs, 458 seed
words, rank ceiling ~457 vs v10's 60). Seed vectors come from the v07/v09/
v12 extractions (same 12-frame protocol). Dead-zone calibration is the v10
protocol verbatim, run on the v09 vocabulary at layer 12: per axis, the
smallest tau on the z-grid with >= 95% split-half sign agreement among
committed words at >= 10% coverage; uncertifiable axes are dropped. Ordering
is greedy geometric decorrelation. Everything is frozen before benchmark
vectors are touched.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for v in ("version_07_prime_basis", "version_09_semantic_barcode"):
    sys.path.insert(0, os.path.join(HERE, "..", v))

from pairs import ALL_PAIRS, PAIR_NAMES, seed_words  # noqa: E402

LAYER = 12
AGREE_MIN = 0.95
COVERAGE_MIN = 0.10
TAU_GRID = np.round(np.arange(0.0, 2.01, 0.05), 2)
BANK_NPZ = os.path.join(HERE, "wide_bank.npz")

V07_NPZ = os.path.join(HERE, "..", "version_07_prime_basis", "vectors.npz")
V09_NPZ = os.path.join(HERE, "..", "version_09_semantic_barcode", "vocab_vectors.npz")
SEED_NPZ = os.path.join(HERE, "seed_vectors.npz")
BENCH_NPZ = os.path.join(HERE, "..", "version_11_benchmark", "bench_vectors.npz")


def _pooled(npz, layer=LAYER):
    z = np.load(npz, allow_pickle=True)
    words = [str(w) for w in z["words"]]
    splits = [str(s) for s in z["split_names"]]
    means = z["sums"] / z["counts"][None, :, None, None]
    ia, ib = splits.index("half_a"), splits.index("half_b")
    return words, means[:, ia, layer, :], means[:, ib, layer, :]


def seed_vector_lut():
    lut = {}
    for npz in (V07_NPZ, V09_NPZ, SEED_NPZ):
        words, a, b = _pooled(npz)
        for i, w in enumerate(words):
            lut.setdefault(w, (a[i] + b[i]) / 2)
    missing = [w for w in seed_words() if w not in lut]
    assert not missing, f"seeds missing vectors: {missing}"
    return lut


def build_bank():
    lut = seed_vector_lut()
    A = np.stack([lut[a] - lut[b] for a, b in ALL_PAIRS])
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)

    # calibration domain: v09 vocabulary, frozen transform within-domain
    words, ha, hb = _pooled(V09_NPZ)
    pooled = (ha + hb) / 2
    g09 = pooled.mean(0)

    def tf(V):
        V = V - g09
        return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

    Vp, Va, Vb = tf(pooled), tf(ha), tf(hb)
    proj = Vp @ A.T
    mu, sd = proj.mean(0), proj.std(0) + 1e-9
    z = (proj - mu) / sd
    agree = np.sign(Va @ A.T) == np.sign(Vb @ A.T)

    keep, taus, covs = [], [], []
    for i in range(len(ALL_PAIRS)):
        for tau in TAU_GRID:
            committed = np.abs(z[:, i]) >= tau
            if committed.mean() < COVERAGE_MIN:
                break
            if agree[committed, i].mean() >= AGREE_MIN:
                keep.append(i); taus.append(tau); covs.append(committed.mean())
                break

    keep = np.array(keep)
    Ak = A[keep]
    order = [0]
    mc = np.abs(Ak @ Ak[0])
    for _ in range(len(keep) - 1):
        mc[order] = np.inf
        nxt = int(np.argmin(mc))
        order.append(nxt)
        mc = np.maximum(mc, np.abs(Ak @ Ak[nxt]))
    o = np.array(order)

    np.savez_compressed(
        BANK_NPZ, axes=Ak[o], names=np.array([PAIR_NAMES[i] for i in keep[o]]),
        tau=np.array(taus)[o], coverage=np.array(covs)[o],
        mu=mu[keep[o]], sd=sd[keep[o]], g09=g09, layer=LAYER)
    return len(keep), len(ALL_PAIRS)


def load_bank():
    z = np.load(BANK_NPZ, allow_pickle=True)
    return {k: z[k] for k in z.files}


def bench_vectors(bank):
    words, ha, hb = _pooled(BENCH_NPZ)
    V = (ha + hb) / 2 - bank["g09"]
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return words, V


def ternary_codes(V, bank, K):
    z = (V @ bank["axes"][:K].T - bank["mu"][:K]) / bank["sd"][:K]
    return (np.sign(z) * (np.abs(z) >= bank["tau"][:K])).astype(np.int8)


def code_cos_rows(C, i, j):
    a, b = C[i].astype(np.float32), C[j].astype(np.float32)
    return float((a @ b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))
