"""step2 — H3: do the 13 prime axes preserve target-word geometry?

Full-space ground truth: cosine similarity among the 100 target vectors
(896-dim). Compressed view: cosine among 13-dim projection profiles
(z-scored per axis). Statistic: Spearman rho between the two similarity
matrices' upper triangles, plus top-10 nearest-neighbor Jaccard overlap.

Compared against TWO nulls (200 resamples each) and one ceiling:
  null A  random control-pool pairs   — are prime WORDS special?
  null B  shuffled prime pairings     — is the NSM PAIRING special?
           (same 26 prime words, re-paired at random)
  ceiling PCA(13) fit on the target vectors — the optimal 13-dim linear view.

Pre-registered pass for H3: p < 0.05 against BOTH nulls.
"""
import numpy as np
from scipy.stats import spearmanr

from axes import AXES
from step1_project import load_all, project

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "version_07_prime_basis"))
from concepts import CONTROL_POOL, TARGETS  # noqa: E402

N_NULL = 200


def zscore(P):
    return (P - P.mean(0)) / (P.std(0) + 1e-9)


def sim_upper(V):
    U = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    S = U @ U.T
    return S[np.triu_indices(len(V), k=1)]


def knn_jaccard(V_full, V_low, k=10):
    def topk(V):
        U = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        S = U @ U.T
        np.fill_diagonal(S, -np.inf)
        return np.argsort(-S, axis=1)[:, :k]
    A, B = topk(V_full), topk(V_low)
    j = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(A, B)]
    return float(np.mean(j))


def geometry_score(vecs, pairs, V_full, full_upper):
    P = zscore(project(vecs, pairs))
    rho = spearmanr(full_upper, sim_upper(P)).statistic
    return float(rho), knn_jaccard(V_full, P)


def main():
    vecs, layer, rel = load_all()
    V_full = np.stack([vecs[w] for w in TARGETS])
    full_upper = sim_upper(V_full)
    rng = np.random.default_rng(7)

    rho_p, jac_p = geometry_score(vecs, AXES, V_full, full_upper)
    print(f"layer {layer} | prime axes: rho={rho_p:.4f}  top-10 jaccard={jac_p:.4f}\n")

    # null A: random control-pool pairs
    nullA = []
    for _ in range(N_NULL):
        idx = rng.choice(len(CONTROL_POOL), size=2 * len(AXES), replace=False)
        pairs = [(CONTROL_POOL[idx[2 * i]], CONTROL_POOL[idx[2 * i + 1]]) for i in range(len(AXES))]
        nullA.append(geometry_score(vecs, pairs, V_full, full_upper)[0])
    nullA = np.array(nullA)
    pA = float((nullA >= rho_p).mean())
    print(f"null A (random control pairs): rho={nullA.mean():.4f} +/- {nullA.std():.4f}  p={pA:.3f}")

    # null B: same 26 prime words, shuffled pairing
    prime_words = [w for pair in AXES for w in pair]
    nullB = []
    for _ in range(N_NULL):
        perm = rng.permutation(prime_words)
        pairs = [(perm[2 * i], perm[2 * i + 1]) for i in range(len(AXES))]
        nullB.append(geometry_score(vecs, pairs, V_full, full_upper)[0])
    nullB = np.array(nullB)
    pB = float((nullB >= rho_p).mean())
    print(f"null B (shuffled prime pairs): rho={nullB.mean():.4f} +/- {nullB.std():.4f}  p={pB:.3f}")

    # ceiling: PCA(13) on the target vectors themselves
    Vc = V_full - V_full.mean(0)
    _, _, Vt = np.linalg.svd(Vc, full_matrices=False)
    scores = zscore(Vc @ Vt[:len(AXES)].T)
    rho_pca = spearmanr(full_upper, sim_upper(scores)).statistic
    print(f"ceiling PCA(13) fit on targets: rho={rho_pca:.4f}")
    print(f"\nprime axes reach {rho_p / rho_pca:.0%} of the PCA ceiling")
    print(f"H3 {'PASS' if pA < 0.05 and pB < 0.05 else 'FAIL'} (need p<0.05 vs both nulls)")
    return rho_p, pA, pB


if __name__ == "__main__":
    main()
