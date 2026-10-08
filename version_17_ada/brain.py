"""
BRAIN — the layers are regions; reasoning is signals flowing between them
=========================================================================
Not a search. Four regions — grammar, sentence, idiom, meaning — each its
own map of the words, each owning the pathways of its segments' roles.

  pathways   DIRECTED, like axons: a lit word sends its energy back to the
             words whose glyph holds it (value -> holder), SPLIT across all
             its outgoing threads — a hub's energy spreads thin, a specific
             word's arrives strong.  P_L = A_Lᵀ-direction, column-normalized.
  flow       time steps; each step every region fires on the current signal,
             and the signal becomes ONLY what fired this step (not a running
             sum — a running sum is an OR: every one-hop neighbour of any
             clue keeps glowing and outshines the answer).
  senses     each clue enters as its own stream.
  integration where streams converge, a word fires on COINCIDENCE: its score
             is Σ_clue log(ε + arrival) — a product, so it must be reached by
             every clue (ε is the floor for a missing stream, chosen on the
             TRAIN riddles only).

Measured on second-order riddles (reason.py: clues two hops from the answer,
never in its glyph), against exhaustive two-step search.

  why each piece is there (measured on the way, scratch prototypes):
    symmetric pathways (energy also flows to words that MENTION a clue):
        solvable top-1 0%, median rank 46
    directed + energy split, summed:                    solvable 25%, rank 2
    directed + energy split + coincidence:              solvable 100%

Run:  python brain.py
"""

import numpy as np

import reason as R
import think as T
from glyph_schema3 import PATHS

LAYERS = ["grammar", "sentence", "idiom", "meaning"]


class Brain:
    def __init__(self, base=None):
        self.b = b = base or T.Brain()
        self.n = len(b.words)
        self.P = {}
        for L in LAYERS:
            A = sum(b.A[k] for k, p in enumerate(PATHS) if p.startswith(L + "."))
            out = A.sum(0)                                     # value v's outgoing threads
            self.P[L] = A / np.where(out > 0, out, 1)[None, :]  # holder <- value, energy split
        self.eps = 1e-6
        self._streams = {}

    def stream(self, clue, steps=2):
        """One sense stream: per step, per region, what fired. (Cached: a stream
        depends only on the clue word.)"""
        if (clue, steps) in self._streams:
            return self._streams[clue, steps]
        x = np.zeros(self.b.N); x[self.b.idx[clue]] = 1.0
        trace = []
        for _ in range(steps):
            fired = {L: self.P[L] @ x for L in LAYERS}
            trace.append(fired)
            x = sum(fired.values())                            # only this step's firing
        self._streams[clue, steps] = (x[:self.n], trace)
        return self._streams[clue, steps]

    def think(self, clues, steps=2):
        streams = [self.stream(c, steps) for c in clues]
        score = np.log(np.stack([s for s, _ in streams]) + self.eps).sum(0)
        for c in clues:
            if self.b.idx[c] < self.n:
                score[self.b.idx[c]] = -np.inf
        return score, streams

    def ranks(self, qs):
        return np.array([R.rank_of(self.think(cl)[0], self.b.idx[t]) for t, cl, _ in qs])

    def explain(self, clues, word):
        """Which regions carried each clue's signal into `word` on the last step."""
        _, streams = self.think(clues)
        i = self.b.idx[word]
        out = []
        for c, (_, trace) in zip(clues, streams):
            last = trace[-1]
            tot = sum(last[L][i] for L in LAYERS)
            out.append((c, {L: last[L][i] / tot for L in LAYERS if tot and last[L][i] > 0}))
        return out


if __name__ == "__main__":
    base = T.Brain()
    brain = Brain(base)
    rng = np.random.default_rng(0)
    order = rng.permutation(base.words)
    split = int(0.7 * len(order))
    train_w, test_w = list(order[:split]), list(order[split:])
    qtr = R.make_riddles(base, train_w, seed=1, per_word=10)
    qte = R.make_riddles(base, test_w, seed=2, per_word=10)

    best = None                                                # ε chosen on TRAIN only
    for eps in (1e-8, 1e-6, 1e-4, 1e-2):
        brain.eps = eps
        acc = np.mean(brain.ranks(qtr) < 0.5)
        print(f"train riddles, ε={eps:g}: top-1 {acc:.1%}")
        if best is None or acc > best[1]:
            best = (eps, acc)
    brain.eps = best[0]
    print(f"-> ε = {brain.eps:g}\n")

    deg = R.in_degree(base)
    l2 = R.lookup(base, qte, 2)
    rb = brain.ranks(qte)
    buckets = [("rare (0-1)", (0, 1)), ("sparse (2-4)", (2, 4)), ("common (5+)", (5, 10 ** 6))]
    print(f"test: {len(qte)} second-order riddles about {len(test_w)} never-practised words")
    print("top-1 / top-5")
    print(f"  {'':28}{'all':>14}" + "".join(f"{b:>16}" for b, _ in buckets))
    for name, r in (("two-step search (explicit)", l2), ("brain: 4 regions, flowing", rb)):
        cells = f"{np.mean(r < .5):6.1%} /{np.mean(r < 5):6.1%}"
        for _, (lo, hi) in buckets:
            m = np.array([lo <= deg[t] <= hi for t, _, _ in qte])
            cells += f"  {np.mean(r[m] < .5):6.1%} /{np.mean(r[m] < 5):6.1%}"
        print(f"  {name:28}{cells}")
    solv = l2 == 0
    print(f"\non the {solv.sum()} riddles explicit search solves uniquely: brain top-1 "
          f"{np.mean(rb[solv] < .5):.1%}")

    print("\nthoughts (clue <- via neighbour; regions that carried each clue into the answer)")
    shown = 0
    for t, clues, ns in qte:
        sc, _ = brain.think(clues)
        got = base.words[int(np.argmax(sc))]
        if got != t:
            continue
        print(f"  {', '.join(f'{c}<-{n}' for c, n in zip(clues, ns))}  =>  {got}")
        for c, regs in brain.explain(clues, got):
            print(f"      {c:12} " + "  ".join(f"{L} {f:.0%}" for L, f in
                                             sorted(regs.items(), key=lambda kv: -kv[1])))
        shown += 1
        if shown == 4:
            break
