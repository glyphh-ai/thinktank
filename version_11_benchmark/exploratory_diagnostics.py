"""EXPLORATORY (not pre-registered) — the v11 autopsy, three diagnostics.

1. Recalibrating mu/sigma on the benchmark vocabulary does NOT restore
   retention (0.337 vs frozen 0.350 on SimLex) — the freeze was not the
   failure; frozen z-drift is small (median 0.16).
2. The good-bad meter vs Warriner valence peaks at LAYER 12 (rho 0.567;
   layer 5 = 0.490, layer 18 = 0.456). Split-half reliability (which picked
   layer 5) is not validity. Meters should be certified at semantic depth.
3. Rebuilding the whole bank at layer 12 does not fix the code (66%
   retention at K=256), and using ALL 1830 axes makes it WORSE (61%).

Root cause of 2+3: every axis is a difference of the same 61 prime vectors,
so the entire bank — any K, any layer — spans a subspace of rank <= 60. The
barcode cannot see the ~30% of human-relevant geometry living outside it;
SimHash-256 samples 256 independent directions and keeps 86%. The fix is not
more pairs from the same words; it is MORE SEED WORDS (named axes from a few
hundred contrast pairs -> rank in the hundreds). That is v12.
"""
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "version_10_ternary_barcode"))
from ternary import calibrate  # noqa: E402

from bench_vocab import benchmark_pairs, warriner  # noqa: E402
from frozen import V09_NPZ, _pooled, code_cos, load_frozen, bench_vectors  # noqa: E402


def bench_rho(C, wi, name):
    pairs = benchmark_pairs(name)
    return spearmanr(
        [s for _, _, s in pairs],
        [float(code_cos(C[wi[a]:wi[a] + 1], C[wi[b]:wi[b] + 1])[0]) for a, b, _ in pairs],
    ).statistic


def main():
    bank, layer, g09 = load_frozen()
    words, V = bench_vectors(layer, g09)
    wi = {w: i for i, w in enumerate(words)}
    K = 256
    A = bank["axes"][:K]
    proj = V @ A.T

    print("1) frozen vs recalibrated mu/sigma (tau frozen), K=256:")
    for label, z in (("frozen", (proj - bank["mu"][:K]) / bank["sd"][:K]),
                     ("recalibrated", (proj - proj.mean(0)) / (proj.std(0) + 1e-9))):
        C = (np.sign(z) * (np.abs(z) >= bank["tau"][:K])).astype(np.int8)
        print(f"   {label:<13s} SimLex {bench_rho(C, wi, 'simlex.txt'):.4f} "
              f"WS353 {bench_rho(C, wi, 'ws353.txt'):.4f}")

    print("\n2) good-bad meter vs Warriner valence by layer:")
    val = warriner()
    w07, s07, m07 = _means_v07()
    zB = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_vectors.npz"),
                 allow_pickle=True)
    wB = [str(w) for w in zB["words"]]
    mB = zB["sums"] / zB["counts"][None, :, None, None]
    idx = [i for i, w in enumerate(wB) if w in val]
    y = np.array([val[wB[i]] for i in idx])
    for L in range(0, m07.shape[2], 3):
        P07 = (m07[:, 0, L, :] + m07[:, 1, L, :]) / 2
        lut = {w: P07[i] for i, w in enumerate(w07)}
        ax = lut["good"] - lut["bad"]
        ax = ax / np.linalg.norm(ax)
        VB = (mB[:, 0, L, :] + mB[:, 1, L, :]) / 2
        print(f"   layer {L:>2d}: rho {spearmanr(y, VB[idx] @ ax).statistic:.4f}")

    print("\n3) bank rebuilt at layer 12, frozen-from-v09:")
    bank12 = calibrate(12)
    _, V09_raw = _pooled(V09_NPZ, 12)
    words12, raw = _pooled(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "bench_vectors.npz"), 12)
    V12 = raw - V09_raw.mean(0)
    V12 = V12 / (np.linalg.norm(V12, axis=1, keepdims=True) + 1e-9)
    wi12 = {w: i for i, w in enumerate(words12)}
    for Kx in (256, len(bank12["names"])):
        z = (V12 @ bank12["axes"][:Kx].T - bank12["mu"][:Kx]) / bank12["sd"][:Kx]
        C = (np.sign(z) * (np.abs(z) >= bank12["tau"][:Kx])).astype(np.int8)
        print(f"   K={Kx:>4d}: SimLex {bench_rho(C, wi12, 'simlex.txt'):.4f} "
              f"WS353 {bench_rho(C, wi12, 'ws353.txt'):.4f}")
    print(f"\n   rank ceiling: all axes are differences of 61 prime vectors -> rank <= 60.")


def _means_v07():
    z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "version_07_prime_basis", "vectors.npz"), allow_pickle=True)
    return ([str(w) for w in z["words"]], [str(s) for s in z["split_names"]],
            z["sums"] / z["counts"][None, :, None, None])


if __name__ == "__main__":
    main()
