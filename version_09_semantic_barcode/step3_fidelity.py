"""step3 — THE fidelity-tax curve: named bits vs SimHash at equal K.

Ground truth: full 896-dim cosine geometry of the 1027 eval words. For each
code width K, compare Spearman rho (code agreement vs full cosine, all ~527k
word pairs) and top-10 neighbor Jaccard for:

  NAMED    first K axes of the certified prime-pair bank (nested family)
  SIMHASH  K random Gaussian hyperplanes — the LSH optimum (5 seeds)
  WORDPAIR K random control-pool word-pair axes — is pair-ness enough,
           or is prime-ness doing anything? (5 resamples)

The fidelity tax at K is 1 - rho_named/rho_simhash.

Pre-registered K2: at K = 256, named rho >= 0.80 * simhash rho.
(v08 predicts named loses at small K — thin blades; the question is whether
width buys the tax down to something interpretability is worth.)
"""
import numpy as np
from scipy.stats import spearmanr

from barcode import (CONTROL_POOL, code_sim_upper, codes, cos_sim_upper,
                     eval_vectors, load_axis_bank, pair_axes, word_vectors_v07)

KS = [16, 32, 64, 128, 256, 512, 1024]
N_SEEDS = 5


def knn_jaccard_from_sims(S_full, S_code, k=10):
    def topk(S):
        S = S.copy()
        np.fill_diagonal(S, -np.inf)
        return np.argsort(-S, axis=1)[:, :k]
    A, B = topk(S_full), topk(S_code)
    return float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(A, B)]))


def full_sim(V):
    U = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return U @ U.T


def evaluate(V, axes, S_full, full_upper):
    C = codes(V, axes)
    rho = spearmanr(full_upper, code_sim_upper(C)).statistic
    Sc = (C.astype(np.float32) @ C.T.astype(np.float32)) / C.shape[1]
    return float(rho), knn_jaccard_from_sims(S_full, Sc)


def main():
    bank, names, stability, layer = load_axis_bank()
    words, V = eval_vectors(layer)
    S_full = full_sim(V)
    full_upper = cos_sim_upper(V)
    d = V.shape[1]
    rng = np.random.default_rng(7)

    ctrl_vecs = word_vectors_v07(layer, CONTROL_POOL)
    results = {}
    print(f"layer {layer}, {len(words)} words, bank of {len(bank)} named axes\n")
    print(f"{'K':>5s} {'named rho':>10s} {'simhash rho':>12s} {'wordpair rho':>13s} "
          f"{'tax':>7s} {'named jac':>10s} {'simhash jac':>12s}")
    for K in KS:
        if K > len(bank):
            print(f"{K:>5d}  (bank exhausted at {len(bank)})")
            continue
        rho_n, jac_n = evaluate(V, bank[:K], S_full, full_upper)

        sh_r, sh_j = [], []
        for _ in range(N_SEEDS):
            H = rng.normal(size=(K, d))
            H /= np.linalg.norm(H, axis=1, keepdims=True)
            r, j = evaluate(V, H, S_full, full_upper)
            sh_r.append(r); sh_j.append(j)

        wp_r = []
        for _ in range(N_SEEDS):
            # sample K pairs of distinct control words (with replacement across pairs)
            wp = []
            while len(wp) < K:
                i, j = rng.choice(len(CONTROL_POOL), size=2, replace=False)
                wp.append((i, j))
            wp_r.append(evaluate(V, pair_axes(ctrl_vecs, wp), S_full, full_upper)[0])

        tax = 1 - rho_n / np.mean(sh_r)
        results[K] = dict(named=rho_n, simhash=float(np.mean(sh_r)),
                          wordpair=float(np.mean(wp_r)), tax=float(tax),
                          jac_named=jac_n, jac_simhash=float(np.mean(sh_j)))
        print(f"{K:>5d} {rho_n:>10.4f} {np.mean(sh_r):>12.4f} {np.mean(wp_r):>13.4f} "
              f"{tax:>6.1%} {jac_n:>10.4f} {np.mean(sh_j):>12.4f}")

    if 256 in results:
        r = results[256]
        ok = r["named"] >= 0.80 * r["simhash"]
        print(f"\nK2 at K=256: named {r['named']:.4f} vs 0.80*simhash "
              f"{0.80 * r['simhash']:.4f} -> {'PASS' if ok else 'FAIL'} "
              f"(tax {r['tax']:.1%})")
        return ok, results
    return False, results


if __name__ == "__main__":
    main()
