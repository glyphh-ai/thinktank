"""EXPLORATORY (not pre-registered): rerun the v09 pipeline at layer 18.

The blind layer criterion picked layer 5 (reliability 0.915) over layer 18
(0.908) — a near-tie. v07/v08 ran at 18, and the layer-5 composition
neighborhoods looked register-dominated (film/jobs/videos). This script asks
whether the pre-registered verdict is layer-luck: same bank rules, fidelity
at K=256, and the same 15 probes, with the layer forced to 18. Results go in
FINDINGS.md clearly labeled exploratory; they do NOT override the
pre-registered layer-5 verdict.
"""
import numpy as np
from scipy.stats import spearmanr

import step4_compose
from barcode import (PRIMES, all_prime_pairs, code_sim_upper, codes,
                     cos_sim_upper, eval_vectors, pair_axes, word_vectors_v07)
from vocab import PROBES

LAYER = 18
K = 256


def mine(layer):
    P = word_vectors_v07(layer, PRIMES)
    pairs = all_prime_pairs()
    A = pair_axes(P, pairs)
    names = [f"{PRIMES[i]}-{PRIMES[j]}" for i, j in pairs]
    _, Va = eval_vectors(layer, split="half_a")
    _, Vb = eval_vectors(layer, split="half_b")
    st = (np.sign(Va @ A.T) == np.sign(Vb @ A.T)).mean(0)
    keep = np.where(st >= 0.9)[0]
    Ak = A[keep]
    order = [int(np.argmax(st[keep]))]
    mc = np.abs(Ak @ Ak[order[0]])
    for _ in range(len(keep) - 1):
        mc[order] = np.inf
        nxt = int(np.argmin(mc))
        order.append(nxt)
        mc = np.maximum(mc, np.abs(Ak @ Ak[nxt]))
    idx = keep[np.array(order)]
    return A[idx], [names[i] for i in idx]


def main():
    bank, names = mine(LAYER)
    words, V = eval_vectors(LAYER)
    print(f"EXPLORATORY layer {LAYER}: {len(bank)} stable axes")

    full_upper = cos_sim_upper(V)
    C = codes(V, bank[:K])
    rho_n = spearmanr(full_upper, code_sim_upper(C)).statistic
    rng = np.random.default_rng(7)
    sh = []
    for _ in range(5):
        H = rng.normal(size=(K, V.shape[1]))
        H /= np.linalg.norm(H, axis=1, keepdims=True)
        sh.append(spearmanr(full_upper, code_sim_upper(codes(V, H))).statistic)
    print(f"K={K}: named rho={rho_n:.4f} simhash rho={np.mean(sh):.4f} "
          f"tax={1 - rho_n / np.mean(sh):.1%}")

    wi = {w: i for i, w in enumerate(words)}
    step4_compose.TIE = codes(np.random.default_rng(3).normal(size=(1, V.shape[1])), bank[:K])[0]
    hits = 0
    for a, b, exp in PROBES:
        r, order = step4_compose.rank_of_expected(C, wi, wi[a], wi[b], [wi[e] for e in exp])
        hit = r <= step4_compose.TOP
        hits += hit
        top5 = ", ".join(words[o] for o in order[:5])
        print(f"{'HIT ' if hit else 'miss'} {a}+{b} -> {exp} best rank {r:>4d} | top5: {top5}")
    print(f"{hits}/{len(PROBES)} probes hit top-{step4_compose.TOP} at layer {LAYER}")


if __name__ == "__main__":
    main()
