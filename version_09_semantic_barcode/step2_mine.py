"""step2 — mine and certify the named axis bank.

Candidates: all C(61,2) = 1830 prime-pair difference axes. Certification and
ordering use NO downstream test signal:

  stability  — per axis, sign-agreement between half-A and half-B projections
               across the 1027 eval words. Keep axes with agreement >= 0.9.
               (K1: at least 256 must survive, else the barcode dies at width.)
  ordering   — greedy farthest-point traversal on |cos| between the axis
               vectors THEMSELVES (896-dim), seeded at the most stable axis.
               Purely geometric decorrelation: no eval-vocabulary data is used
               for the ordering, so the named bank gets no adaptive advantage
               over SimHash in step3.

Output: axes.npz (axis vectors, names like "good-bad", stability, layer) —
the first K rows are "the K-bit named codebook" for every downstream step.
"""
import numpy as np

from barcode import (PRIMES, all_prime_pairs, best_layer, eval_vectors,
                     pair_axes, word_vectors_v07, AXES_NPZ)

STABILITY_MIN = 0.90
K1_MIN_AXES = 256


def main():
    layer, rel = best_layer()
    print(f"layer {layer} (eval split-half reliability {rel:.3f})")

    P = word_vectors_v07(layer, PRIMES)
    pairs = all_prime_pairs()
    A = pair_axes(P, pairs)
    names = [f"{PRIMES[i]}-{PRIMES[j]}" for i, j in pairs]

    _, Va = eval_vectors(layer, split="half_a")
    _, Vb = eval_vectors(layer, split="half_b")
    sa, sb = np.sign(Va @ A.T), np.sign(Vb @ A.T)
    stability = (sa == sb).mean(axis=0)

    keep = np.where(stability >= STABILITY_MIN)[0]
    print(f"{len(keep)}/{len(pairs)} axes stable at >= {STABILITY_MIN} "
          f"(K1 needs >= {K1_MIN_AXES}) | median stability {np.median(stability):.3f}")

    # greedy farthest-point ordering by |cos| between axis vectors
    Ak = A[keep]
    order = [int(np.argmax(stability[keep]))]
    max_cos = np.abs(Ak @ Ak[order[0]])
    for _ in range(len(keep) - 1):
        max_cos[order] = np.inf
        nxt = int(np.argmin(max_cos))
        order.append(nxt)
        max_cos = np.maximum(max_cos, np.abs(Ak @ Ak[nxt]))

    idx = keep[np.array(order)]
    np.savez_compressed(AXES_NPZ, axes=A[idx], names=np.array([names[i] for i in idx]),
                        stability=stability[idx], layer=layer)
    print(f"bank: {len(idx)} axes. first 12: {[names[i] for i in idx[:12]]}")
    return len(keep) >= K1_MIN_AXES


if __name__ == "__main__":
    main()
