"""step2 — T1: does the dead zone buy fidelity, bit for bit?

Paired comparison on the SAME axes (first K of the ternary bank): ternary
codes vs plain binary sign codes, Spearman rho against the full 896-dim
cosine geometry (~527k pairs). SimHash at the same K shown for reference.

Pre-registered T1: ternary rho > binary rho at K=256. (The claim under test
is exactly "coin-flip bits from the neutral mass dilute the code".)
"""
import numpy as np
from scipy.stats import spearmanr

from ternary import (binary_codes, code_cos_upper, cos_sim_upper, eval_vectors,
                     load_bank, ternary_codes)

KS = [64, 128, 256, 512]


def main():
    bank = load_bank()
    layer = int(bank["layer"])
    words, V = eval_vectors(layer)
    full_upper = cos_sim_upper(V)
    n_bank = len(bank["axes"])
    rng = np.random.default_rng(7)

    print(f"layer {layer}, bank {n_bank} axes\n"
          f"{'K':>5s} {'ternary':>9s} {'binary':>9s} {'simhash':>9s}")
    res = {}
    for K in KS:
        if K > n_bank:
            print(f"{K:>5d}  (bank exhausted)")
            continue
        rho_t = spearmanr(full_upper, code_cos_upper(ternary_codes(V, bank, K))).statistic
        rho_b = spearmanr(full_upper, code_cos_upper(binary_codes(V, bank, K))).statistic
        sh = []
        for _ in range(5):
            H = rng.normal(size=(K, V.shape[1]))
            H /= np.linalg.norm(H, axis=1, keepdims=True)
            s = np.sign(V @ H.T)
            s[s == 0] = 1
            sh.append(spearmanr(full_upper, code_cos_upper(s.astype(np.int8))).statistic)
        res[K] = (float(rho_t), float(rho_b), float(np.mean(sh)))
        print(f"{K:>5d} {rho_t:>9.4f} {rho_b:>9.4f} {np.mean(sh):>9.4f}")

    if 256 in res:
        t, b, _ = res[256]
        ok = t > b
        print(f"\nT1 {'PASS' if ok else 'FAIL'}: ternary {t:.4f} vs binary {b:.4f} at K=256")
        return ok, res
    return False, res


if __name__ == "__main__":
    main()
