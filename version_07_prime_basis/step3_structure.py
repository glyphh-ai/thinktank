"""step3 — do prime vectors carry NSM's category structure?

NSM groups its primes into categories (mental predicates, quantifiers, space,
time, ...). If the concept vectors encode meaning rather than surface form,
primes in the same category should be more similar to each other than to
primes in other categories.

Test: mean within-category cosine minus mean between-category cosine, against
a permutation null (category labels shuffled over the primes 10,000 times).
Categories with a single member (similarity) contribute no within pairs.
"""
import argparse

import numpy as np

from concepts import PRIME_CATEGORY, PRIMES
from step2_basis import load_vectors


def category_statistic(sim, labels):
    n = len(labels)
    same = np.array([[labels[i] == labels[j] for j in range(n)] for i in range(n)])
    iu = np.triu_indices(n, k=1)
    within = sim[iu][same[iu]]
    between = sim[iu][~same[iu]]
    return within.mean() - between.mean(), within.mean(), between.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="vectors.npz")
    ap.add_argument("--n-perm", type=int, default=10_000)
    args = ap.parse_args()

    vecs, layer, rel, _ = load_vectors(args.vectors)
    print(f"layer {layer} (reliability {rel:.3f})\n")

    V = np.stack([vecs[w] for w in PRIMES])
    sim = V @ V.T
    labels = [PRIME_CATEGORY[w] for w in PRIMES]

    stat, within, between = category_statistic(sim, labels)
    rng = np.random.default_rng(11)
    perm_stats = []
    for _ in range(args.n_perm):
        shuffled = list(labels)
        rng.shuffle(shuffled)
        perm_stats.append(category_statistic(sim, shuffled)[0])
    perm_stats = np.array(perm_stats)
    p = float((perm_stats >= stat).mean())

    print(f"within-category cos  = {within:.4f}")
    print(f"between-category cos = {between:.4f}")
    print(f"statistic (within - between) = {stat:.4f}")
    print(f"permutation null = {perm_stats.mean():.4f} +/- {perm_stats.std():.4f}")
    print(f"p = {p:.4f}  ({args.n_perm} permutations)")

    # most similar prime pairs, for eyeballing
    iu = np.triu_indices(len(PRIMES), k=1)
    order = np.argsort(-sim[iu])
    print("\ntop prime pairs by cosine:")
    for idx in order[:10]:
        i, j = iu[0][idx], iu[1][idx]
        print(f"  {PRIMES[i]:>8s} ~ {PRIMES[j]:<8s} cos={sim[i, j]:.3f}"
              f"   [{labels[i]} / {labels[j]}]")


if __name__ == "__main__":
    main()
