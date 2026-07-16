"""step3 — T2: composition, the rescue shot. Same 15 pre-registered probes
as v09 (vocab.py), same bar (>= 7/15 hit top-20, p < 0.05 vs 200-run null),
new mechanics: ternary bundling.

bundle(a, b) = sign(a + b) — and 0 stays 0, including the +1/-1 conflict
case: where two codes disagree the bundle now ABSTAINS instead of
coin-flipping. Ranking by cosine over ternary codes (sparse-aware).
"""
import numpy as np

from ternary import PROBES, eval_vectors, load_bank, ternary_codes

K = 256
TOP = 20
N_NULL = 200


def bundle2(ca, cb):
    return np.sign(ca.astype(np.int16) + cb.astype(np.int16)).astype(np.int8)


def rank_of_expected(C, a_i, b_i, exp_i):
    b = bundle2(C[a_i], C[b_i]).astype(np.float32)
    Cf = C.astype(np.float32)
    sims = (Cf @ b) / ((np.linalg.norm(Cf, axis=1) + 1e-9) * (np.linalg.norm(b) + 1e-9))
    sims[[a_i, b_i]] = -np.inf
    order = np.argsort(-sims)
    return min(int(np.where(order == e)[0][0]) + 1 for e in exp_i), order


def main():
    bank = load_bank()
    layer = int(bank["layer"])
    words, V = eval_vectors(layer)
    wi = {w: i for i, w in enumerate(words)}
    C = ternary_codes(V, bank, K)

    hits = 0
    print(f"layer {layer}, K={K} ternary bits\n")
    for a, b, exp in PROBES:
        r, order = rank_of_expected(C, wi[a], wi[b], [wi[e] for e in exp])
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
            r, _ = rank_of_expected(C, int(ra), int(rb), [wi[e] for e in exp])
            h += r <= TOP
        null_hits.append(h)
    null_hits = np.array(null_hits)
    p = float((null_hits >= hits).mean())

    print(f"\n{hits}/{len(PROBES)} hit top-{TOP}; null {null_hits.mean():.2f} "
          f"+/- {null_hits.std():.2f}, p={p:.3f}")
    ok = hits >= 7 and p < 0.05
    print(f"T2 {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    main()
