"""
REINFORCE — pathways form from experience, and they differ by mind
==================================================================
The STRUCTURE is shared and fixed: the v3 glyphs and their lattice threads
(hate -opposite-> love AND hate -partner-> love both exist for everyone).
What a MIND learns is which threads carry energy, and how it reacts:

  pathways  W[word, role]  weight of each thread; energy from a word splits
                           across its threads ∝ W · decode strength
  reaction  V[word]        valence: what outcomes this word was active in

An EPISODE: the context words fire together through the mind's pathways
(depth-2 spread), producing activity a(v) on every word reached and the
energy e carried by every thread; then an outcome r arrives (+1 reward,
-1 pain). Learning, local, no backprop:

  ŝ(v) = max(0, â(v) − b̂(v))
        SPECIFIC activity: normalized activity minus the RESTING activity
        b̂ every spread produces (hubs like person/plain/always are reached
        from everywhere, so they earn no credit). Firing words count 1.
  Δ log W[src, role→v] = η · |r| · ŝ_others(v)        for each firing word src
        ASSOCIATION: a thread strengthens when the REST of the episode
        activates its target — not src's own energy (which includes the
        thread's own weight and runs away: measured, one thread took weight
        26 and the rest 0; crediting src's own spread still let the hub
        "person" take ~12). The spread is linear in its sources, so
        act_others = act_total − act(src). Gated by salience |r| — strong
        reactions burn pathways deeper — then each word's weights
        renormalize (mean 1, bounded): pathways COMPETE.
  ΔV[v] = η_v · r · ŝ(v)
        every specifically active word is tagged with the outcome.

Experience, not rules, decides the fork. Two minds, same structure, same
rule, different childhoods:

  A  loved:   love with care/home/family/joy/warm/help  -> reward
              hate with hurt/anger/enemy/punishment      -> pain
  B  abused:  love with hurt/fear/anger/punishment/rude  -> pain
              hate with hurt/anger/enemy/punishment      -> pain

Run:  python reinforce.py
"""

from collections import defaultdict

import numpy as np

from lattice_decode import Lattice

EXPERIENCE = {
    "loved": [(["love"], ["care", "home", "family", "joy", "warm", "help"], +1.0),
              (["hate"], ["hurt", "anger", "enemy", "punishment", "bad"], -1.0)],
    "abused": [(["love"], ["hurt", "fear", "anger", "punishment", "rude"], -1.0),
               (["hate"], ["hurt", "anger", "enemy", "punishment", "bad"], -1.0)],
}


class Mind:
    def __init__(self, L, eta=0.05, eta_v=0.05, depth=2, beta=0.5, eps=1e-5,
                 w_min=0.05, w_max=20.0, rest=None):
        self.L, self.eta, self.eta_v = L, eta, eta_v
        self.depth, self.beta, self.eps = depth, beta, eps
        self.w_min, self.w_max = w_min, w_max
        self.W = defaultdict(lambda: 1.0)          # (word, role_path) -> weight
        self.V = defaultdict(float)                # word -> valence
        self.rest = rest if rest is not None else resting(L, self)

    def specific(self, act, sources):
        amax = max(act.values())
        s = {v: max(0.0, a / amax - self.rest.get(v, 0.0)) for v, a in act.items()}
        for w in sources:
            s[w] = 1.0
        return s

    def spread(self, words):
        """Fire words together. Returns (activity per word, energy per thread)."""
        act, carried = defaultdict(float), defaultdict(float)
        frontier = [(w, 1.0, 0) for w in words]
        while frontier:
            nxt = []
            for src, energy, hop in frontier:
                threads = self.L.fire(src)
                if not threads:
                    continue
                share = np.array([self.W[src, p] * s for p, _, s in threads])
                share /= share.sum()
                for (p, v, _), sh in zip(threads, share):
                    e = energy * self.beta * sh
                    if e < self.eps:
                        continue
                    carried[src, p, v] += e
                    act[v] += e
                    if hop + 1 < self.depth:
                        nxt.append((v, e, hop + 1))
            frontier = nxt
        return act, carried

    def experience(self, words, r):
        per = {w: self.spread([w])[0] for w in set(words)}       # linear: one spread per source
        act = defaultdict(float)
        for a in per.values():
            for v, x in a.items():
                act[v] += x
        touched = set()
        for src in set(words):
            others = {v: act[v] - per[src].get(v, 0.0) for v in act}
            others = {v: x for v, x in others.items() if x > 1e-12}
            if not others:
                continue
            s_o = self.specific(others, [w for w in words if w != src])
            for p, v, _ in self.L.fire(src):
                dw = self.eta * abs(r) * s_o.get(v, 0.0)
                if dw:
                    self.W[src, p] *= np.exp(dw)
                    touched.add(src)
        s = self.specific(act, words)
        for src in touched:                         # competition: mean weight stays 1
            paths = [p for p, _, _ in self.L.fire(src)]
            m = np.mean([self.W[src, p] for p in paths])
            for p in paths:
                self.W[src, p] = float(np.clip(self.W[src, p] / m, self.w_min, self.w_max))
        for v, sv in s.items():
            self.V[v] += self.eta_v * r * sv

    def field(self, w):
        act, _ = self.spread([w])
        return act

    def valence(self, w):
        return float(np.tanh(self.V[w]))


def resting(L, mind, sample=150):
    """Mean normalized activity of single-word spreads: what fires whatever the input."""
    rng = np.random.default_rng(1)
    words = rng.choice(L.space.words, size=min(sample, len(L.space.words)), replace=False)
    tot = defaultdict(float)
    for w in words:
        act, _ = mind.spread([w])
        amax = max(act.values())
        for v, a in act.items():
            tot[v] += a / amax
    return {v: t / len(words) for v, t in tot.items()}


def cos(a, b):
    k = a.keys() | b.keys()
    x = np.array([a.get(i, 0) for i in k]); y = np.array([b.get(i, 0) for i in k])
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def raise_mind(L, kind, episodes=300, seed=0):
    rng = np.random.default_rng(seed)
    mind = Mind(L, rest=REST)
    for _ in range(episodes):
        anchor, pool, r = EXPERIENCE[kind][rng.integers(len(EXPERIENCE[kind]))]
        ctx = list(rng.choice(pool, size=3, replace=False))
        mind.experience(anchor + ctx, r)
    return mind


def describe(L, mind, label):
    love, hate = mind.field("love"), mind.field("hate")
    watch = ["good", "bad", "care", "hurt", "joy", "anger", "hate", "love", "fear", "helpless"]
    print(f"\n=== {label} ===")
    print(f"  valence      love {mind.valence('love'):+.2f}   hate {mind.valence('hate'):+.2f}"
          f"   care {mind.valence('care'):+.2f}   hurt {mind.valence('hurt'):+.2f}")
    print(f"  love vs hate field similarity: {cos(love, hate):.2f}")
    print("  where love's energy goes:  " +
          ", ".join(f"{v} {love.get(v, 0):.3f}" for v in watch if v != "love"))
    th = sorted(L.fire("love"), key=lambda t: -mind.W["love", t[0]])
    print("  love's strongest pathways: " +
          ", ".join(f"{p.split('.')[-1]}->{v} {mind.W['love', p]:.2f}" for p, v, _ in th[:6]))
    print("  love's weakest pathways:   " +
          ", ".join(f"{p.split('.')[-1]}->{v} {mind.W['love', p]:.2f}" for p, v, _ in th[-4:]))
    hp = {p.split(".")[-1]: (v, mind.W["hate", p]) for p, v, _ in L.fire("hate")}
    print("  hate's threads to love:    " +
          ", ".join(f"{r}->{v} {w:.2f}" for r, (v, w) in hp.items() if v == "love"))
    return love, hate


if __name__ == "__main__":
    L = Lattice(glyphs="glyphs3.jsonl")
    newborn = Mind(L)
    REST = newborn.rest
    describe(L, newborn, "NEWBORN (no experience: every pathway 1.0)")
    for kind in ("loved", "abused"):
        describe(L, raise_mind(L, kind), f"{kind.upper()} child (300 episodes)")
