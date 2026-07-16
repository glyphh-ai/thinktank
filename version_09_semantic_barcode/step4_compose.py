"""step4 — composition probes: does bundling barcodes retrieve concepts?

For each pre-registered probe (a, b, expected-any-of) from vocab.py:
bundle the K=256 named barcodes of a and b (majority vote per bit, fixed
random tie-break), rank the whole vocabulary (minus a, b) by code agreement
with the bundle, and check whether any expected word lands in the top 20 of
~1025.

Null: 200 runs with each probe's (a, b) replaced by random vocab words
(expected lists unchanged). Pre-registered K3: >= 7/15 probes hit AND
empirical p < 0.05 vs the null distribution of hit counts.

v07 killed bundling for *reconstruction*; this asks the easier retrieval
question — is the intersection of two codes' agreements enough to point at a
lexicalized concept?
"""
import numpy as np

from barcode import code_sim_upper, codes, eval_vectors, load_axis_bank  # noqa: F401
from vocab import PROBES

K = 256
TOP = 20
N_NULL = 200


def bundle2(ca, cb, tie):
    s = ca.astype(np.int16) + cb.astype(np.int16)
    out = np.sign(s).astype(np.int8)
    z = out == 0
    out[z] = tie[z]
    return out


def rank_of_expected(C, wi, a_i, b_i, exp_i):
    bundle = bundle2(C[a_i], C[b_i], tie=TIE)
    sims = (C.astype(np.float32) @ bundle.astype(np.float32)) / C.shape[1]
    sims[[a_i, b_i]] = -np.inf
    order = np.argsort(-sims)
    ranks = {int(np.where(order == e)[0][0]) + 1 for e in exp_i}
    return min(ranks), order


def main():
    bank, names, stability, layer = load_axis_bank()
    words, V = eval_vectors(layer)
    wi = {w: i for i, w in enumerate(words)}
    C = codes(V, bank[:K])

    global TIE
    TIE = codes(np.random.default_rng(3).normal(size=(1, V.shape[1])), bank[:K])[0]

    hits = 0
    print(f"layer {layer}, K={K} named bits, vocab {len(words)}\n")
    for a, b, exp in PROBES:
        r, order = rank_of_expected(C, wi, wi[a], wi[b], [wi[e] for e in exp])
        hit = r <= TOP
        hits += hit
        top5 = ", ".join(words[o] for o in order[:5])
        print(f"{'HIT ' if hit else 'miss'} {a}+{b} -> {exp} best rank {r:>4d} | top5: {top5}")

    rng = np.random.default_rng(11)
    null_hits = []
    for _ in range(N_NULL):
        h = 0
        for a, b, exp in PROBES:
            ra, rb = rng.choice(len(words), size=2, replace=False)
            r, _ = rank_of_expected(C, wi, int(ra), int(rb), [wi[e] for e in exp])
            h += r <= TOP
        null_hits.append(h)
    null_hits = np.array(null_hits)
    p = float((null_hits >= hits).mean())

    print(f"\n{hits}/{len(PROBES)} probes hit top-{TOP}; "
          f"null {null_hits.mean():.2f} +/- {null_hits.std():.2f} hits, p={p:.3f}")
    ok = hits >= 7 and p < 0.05
    print(f"K3 {'PASS' if ok else 'FAIL'} (need >=7 hits and p<0.05)")
    return ok


if __name__ == "__main__":
    main()
