"""
CONTEXT FIRE — senses derived from energy, not stored as branches
=================================================================
A word keeps ONE glyph, a superposition of everything it can mean ("mean":
partner=say, patient=word ... AND opposite=kind, purpose=hurt ...). No
sense inventory, no branches, no rules. Context decides which threads carry
the energy:

  context field   c = Σ over context words: energy-weighted glyphs of every
                  word their spread reaches (each context word fires first)
  inhibition      the lattice's RESTING field B (mean field of many single
                  words) is projected out of the context and of every value
                  glyph: hubs ("plain", "person", "always") sit in every field,
                  so without this they win every context. Only what is
                  specific to this context is left.
  gated firing    energy(thread) ∝ strength · exp( z )
                  z = resonance standardized over the word's own threads,
                  resonance = cos( m(value) ⟂ B, c ⟂ B ) — no temperature.
                  Everything is in MEANING space m(w) (no identity atom).
  settling        the threads that win feed their values back into the
                  field: c ← c0 + λ · Σ energy(thread) · glyph(value); fire
                  again; repeat until the energy split stops changing.
                  One firing is a reflex; settling is thinking.

The SENSE is whichever threads lit up together. A sense can also rebuild
roles its glyph never stored: the gated spread runs into neighbouring
lattices (opposite -> kind carries 'adjective'), so the word field after the
spread is the reading.

Run:  python context_fire.py
"""

import numpy as np

from lattice_decode import Lattice, LAYERS


_MEAN = {}

def meaning(L, w):
    """Unit MEANING vector m(w) (the role bundle, no identity atom). Resonance must
    live in meaning space: a full glyph is dominated by its identity atom, so it
    resonates only with fields that contain the word itself, not its meaning.
    Leaves (no glyph yet) have no meaning and return None."""
    if w not in _MEAN:
        _MEAN[w] = L.M[w] / np.linalg.norm(L.M[w]) if w in L.M else None
    return _MEAN[w]


def context_field(L, words, depth=2):
    """Each context word fires and spreads; the field is the energy-weighted sum of meanings."""
    c = np.zeros(L.mem.shape[1])
    for w in words:
        c += meaning(L, w)                                         # the word itself, energy 1
        f = L.spread(w, depth=depth)
        for l in LAYERS:
            for v, e in f["words"][l].items():
                m = meaning(L, v)
                if m is not None:
                    c += e * m
    return c


_BASE = {}

def resting_field(L, sample=150):
    """Mean field of single words: the activity present whatever the context."""
    if "B" not in _BASE:
        rng = np.random.default_rng(1)
        words = rng.choice(L.space.words, size=min(sample, len(L.space.words)), replace=False)
        B = sum(context_field(L, [w]) / np.linalg.norm(context_field(L, [w])) for w in words)
        _BASE["B"] = B / np.linalg.norm(B)
    return _BASE["B"]


def inhibit(L, x):
    B = resting_field(L)
    return x - (x @ B) * B


def gated_fire(L, w, c, tau=None):
    """Threads of w with energy ∝ strength · exp(z), z = standardized resonance with c."""
    threads = L.fire(w)
    c = inhibit(L, c)
    c = c / np.linalg.norm(c)
    zero = np.zeros(L.mem.shape[1])
    vals = np.stack([inhibit(L, m) if (m := meaning(L, v)) is not None else zero
                     for _, v, _ in threads])
    res = (vals @ c) / np.maximum(np.linalg.norm(vals, axis=1), 1e-9)
    z = (res - res.mean()) / (res.std() + 1e-9)
    s = np.array([s for _, _, s in threads])
    e = s * np.exp(z)
    e /= e.sum()
    return [(p, v, float(ei), float(ri)) for (p, v, _), ei, ri in zip(threads, e, res)]


def settle(L, w, context, tau=None, lam=0.5, tol=1e-4, max_iter=50):
    c0 = context_field(L, context)
    c = c0.copy()
    prev = None
    for it in range(1, max_iter + 1):
        threads = gated_fire(L, w, c, tau)
        e = np.array([t[2] for t in threads])
        if prev is not None and np.abs(e - prev).sum() < tol:
            break
        prev = e
        fb = sum(ei * m for (_, v, ei, _) in threads if (m := meaning(L, v)) is not None)
        c = c0 + lam * np.linalg.norm(c0) * fb / np.linalg.norm(fb)
    return threads, it, c


def gated_spread(L, w, c, tau=0.05, depth=3, beta=0.5, eps=1e-5):
    """The spread, with every firing gated by the same field: the full reading."""
    reached = {}
    frontier = [(w, 1.0, 0)]
    while frontier:
        nxt = []
        for src, energy, hop in frontier:
            if src not in L.M:
                continue
            for p, v, ei, _ in gated_fire(L, src, c, tau):
                e = energy * beta * ei
                if e < eps:
                    continue
                reached[v] = reached.get(v, 0) + e
                if hop + 1 < depth:
                    nxt.append((v, e, hop + 1))
        frontier = nxt
    return reached


def report(L, w, contexts, watch, top=8):
    print(f"\n{'=' * 70}\n{w}")
    uniform = {p: 1 / len(L.fire(w)) for p, _, _ in L.fire(w)}
    runs = {}
    for name, ctx in contexts.items():
        threads, iters, c = settle(L, w, ctx)
        runs[name] = ({p: (v, e) for p, v, e, _ in threads}, gated_spread(L, w, c))
        print(f"\n  context '{name}' {ctx}  -> settled in {iters} firings")
        for p, v, e, r in sorted(threads, key=lambda t: -t[2])[:top]:
            print(f"     {p.split('.')[-1]:9} -> {v:10} energy {e:.3f}  ({e / uniform[p]:4.1f}x uniform)")
    names = list(contexts)
    a, b = runs[names[0]][0], runs[names[1]][0]
    moved = sorted(a, key=lambda p: -(a[p][1] - b[p][1]))
    print(f"\n  threads that flip most between '{names[0]}' and '{names[1]}':")
    for p in moved[:5] + moved[-5:]:
        print(f"     {p.split('.')[-1]:9} {a[p][0]:10} {a[p][1]:.3f} vs {b[p][1]:.3f}")
    print("\n  rebuilt by the gated spread (energy reaching each word):")
    print("     " + "".join(f"{n:>14}" for n in names))
    for v in watch:
        print(f"  {v:12}" + "".join(f"{runs[n][1].get(v, 0):14.4f}" for n in names))


if __name__ == "__main__":
    L = Lattice(glyphs="glyphs3.jsonl")
    report(L, "mean",
           {"cruel":  ["rude", "enemy", "punishment", "evil"],
            "intend": ["sentence", "understand", "language", "speak"]},
           watch=["adjective", "verb", "kind", "unkind", "hurt", "word", "say", "meaning"])
    report(L, "kind",
           {"type":   ["type", "sort", "category"],
            "caring": ["support", "trust", "heal", "friendly"]},
           watch=["type", "sort", "category", "group", "caring", "gentle", "good", "nice"])
