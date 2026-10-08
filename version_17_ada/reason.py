"""
REASON — second-order riddles on sparse concepts
================================================
think.py's riddles were retrieval: clues came from the answer's own glyph, and
plain lookup (70.7%) beat settling (25.8%). Here retrieval fails BY
CONSTRUCTION. Every clue is two hops from the answer:

    answer t  →  neighbour n (a value in t's glyph, itself glyphed)
              →  clue c (a value in n's glyph, NOT in t's glyph, not t, not n)

3 clues, each through a different neighbour. To answer, the facts of several
glyphs must be combined.

Contestants (all score the 395 glyphed words; clue words are excluded):
  lookup-1   count clues in each word's own glyph          (0 by construction)
  lookup-2   count clues in the glyphs of each word's      (explicit two-step
             values — an exhaustive two-step search         search: the
                                                            deliberate baseline)
  settle     the lattice settles on the bare clues (think.Brain), untrained
             and with θ trained by equilibrium propagation on train words
  settle-AND each clue settles ALONE; fields combine by product of their lift
             over resting activity (coincidence detection). Linear settling
             sums fields — an OR — so words one hop from ANY clue outshine
             the answer, which is two hops from ALL of them.

Sparse concepts: answers bucketed by in-degree — how many OTHER glyphs mention
the word. Rare concepts (mentioned by 0-1 glyphs) are reported separately.

Run:  python reason.py
"""

import numpy as np

import think as T


def out_vals(brain, w):
    return {v for v in brain.G[w].values() if v != w}


def make_riddles(brain, words, k=3, per_word=3, seed=0):
    rng = np.random.default_rng(seed)
    qs = []
    for t in words:
        own = out_vals(brain, t)
        paths = {}                                   # neighbour -> usable clues
        for n in own:
            if n in brain.G and n != t:
                cl = sorted(out_vals(brain, n) - own - {t, n})
                if cl:
                    paths[n] = cl
        if len(paths) < k:
            continue
        for _ in range(per_word):
            ns = rng.choice(sorted(paths), size=k, replace=False)
            clues = [str(rng.choice(paths[n])) for n in ns]
            if len(set(clues)) == k:
                qs.append((t, clues, list(ns)))
    return qs


def to_batch(brain, qs):
    B = np.stack([brain.question(c) for _, c, _ in qs], 1)
    t = np.array([brain.idx[w] for w, _, _ in qs])
    return B, t, [c for _, c, _ in qs]


def rank_of(scores, target):
    """Rank of the target with random tie-breaking (expected rank among equals)."""
    better = (scores > scores[target]).sum()
    ties = (scores == scores[target]).sum() - 1
    return better + ties / 2


def lookup(brain, qs, hops):
    vals1 = [out_vals(brain, w) for w in brain.words]
    vals2 = [set().union(*[out_vals(brain, n) for n in v if n in brain.G]) | v for v in vals1]
    vals = vals1 if hops == 1 else vals2
    ranks = []
    for t, clues, _ in qs:
        s = np.array([len(set(clues) & v) for v in vals], float)
        for c in clues:
            if c in brain.G:
                s[brain.idx[c]] = -1
        ranks.append(rank_of(s, brain.idx[t]))
    return np.array(ranks)


def settle_ranks(brain, qs):
    B, t, ex = to_batch(brain, qs)
    X, steps = brain.settle(B)
    s = brain.answer(X, ex)
    return np.array([rank_of(s[:, j], t[j]) for j in range(len(qs))]), steps


def and_ranks(brain, qs):
    """Coincidence detection: one field per clue, combined by product of lifts."""
    n = len(brain.words)
    Bs = np.stack([brain.question([w]) for w in brain.words], 1)
    rest = brain.settle(Bs)[0][:n].mean(1)                     # resting activity
    ranks = []
    for t, clues, _ in qs:
        X = brain.settle(np.stack([brain.question([c]) for c in clues], 1))[0][:n]
        sc = np.log(np.maximum(X, 0) / (rest[:, None] + 1e-9) + 1e-6).sum(1)
        for c in clues:
            if c in brain.idx and brain.idx[c] < n:
                sc[brain.idx[c]] = -np.inf
        ranks.append(rank_of(sc, brain.idx[t]))
    return np.array(ranks)


def in_degree(brain):
    deg = {w: 0 for w in brain.words}
    for w, g in brain.G.items():
        for v in set(g.values()):
            if v in deg and v != w:
                deg[v] += 1
    return deg


def table(rows, buckets, qs, deg):
    head = "".join(f"{b:>14}" for b, _ in buckets)
    print(f"  {'':22}{'all':>8}{head}")
    for name, ranks in rows:
        cells = f"{np.mean(ranks < 0.5):8.1%}"
        for _, (lo, hi) in buckets:
            m = np.array([lo <= deg[t] <= hi for t, _, _ in qs])
            cells += f"{np.mean(ranks[m] < 0.5):14.1%}" if m.any() else f"{'-':>14}"
        print(f"  {name:22}{cells}")


if __name__ == "__main__":
    brain = T.Brain(c=0.85)
    rng = np.random.default_rng(0)
    order = rng.permutation(brain.words)
    split = int(0.7 * len(order))
    train_w, test_w = list(order[:split]), list(order[split:])
    qtr = make_riddles(brain, train_w, seed=1)
    qte = make_riddles(brain, test_w, seed=2)
    deg = in_degree(brain)
    buckets = [("rare (0-1)", (0, 1)), ("sparse (2-4)", (2, 4)), ("common (5+)", (5, 10 ** 6))]
    counts = {b: sum(lo <= deg[t] <= hi for t, _, _ in qte) for b, (lo, hi) in buckets}
    print(f"second-order riddles: {len(qtr)} train / {len(qte)} test (test answers never "
          f"practised); test answers by in-degree: {counts}")
    print(f"chance: {1 / len(brain.words):.2%}\n")

    l1, l2 = lookup(brain, qte, 1), lookup(brain, qte, 2)
    r0, st0 = settle_ranks(brain, qte)
    ra = and_ranks(brain, qte)
    Btr, ttr, extr = to_batch(brain, qtr)
    brain.train(Btr, ttr, extr, eta=0.3, eta_w=0.0)
    r1, st1 = settle_ranks(brain, qte)

    print("top-1 accuracy on unseen answers")
    table([("lookup-1 (retrieval)", l1), ("lookup-2 (2-step search)", l2),
           ("settle, untrained", r0), ("settle, trained θ", r1),
           ("settle-AND, untrained", ra)], buckets, qte, deg)
    print("\ntop-5")
    for name, r in (("lookup-2", l2), ("settle, untrained", r0), ("settle, trained θ", r1),
                    ("settle-AND, untrained", ra)):
        print(f"  {name:22}{np.mean(r < 5):8.1%}")
    print(f"\nsettle steps: untrained {st0.mean():.1f}, trained {st1.mean():.1f}")
    solv = [q for q, r in zip(qte, l2) if r == 0]
    if solv:
        print(f"\non the {len(solv)} riddles 2-step search solves uniquely:")
        for name, r in (("settle (sum)", settle_ranks(brain, solv)[0]),
                        ("settle-AND", and_ranks(brain, solv))):
            print(f"  {name:22} top-1 {np.mean(r < 0.5):5.1%}  top-5 {np.mean(r < 5):5.1%}  "
                  f"median rank {np.median(r):.0f}")

    print("\nexamples (clue ← via neighbour), test answers:")
    B, t, ex = to_batch(brain, qte[:40])
    X, _ = brain.settle(B)
    s = brain.answer(X, ex)
    shown = 0
    for j, (ans, clues, ns) in enumerate(qte[:40]):
        if j % 4:
            continue
        got = brain.words[int(np.argmax(s[:, j]))]
        via = ", ".join(f"{c}←{n}" for c, n in zip(clues, ns))
        print(f"   {via:46} -> {got:12} {'OK' if got == ans else '(was ' + ans + ')'}")
        shown += 1
        if shown == 8:
            break
