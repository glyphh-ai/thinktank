"""
OPEN ENCODE v2 — keys are glyphs too (glyph_schema2, glyphs2.jsonl)
====================================================================
A glyph is a bundle of key/value binds. v1 keys were random role atoms;
here every layer, segment and role name is a word with its own glyph, so a
key is built from the schema words' meanings:

    K(l.s.r) = sign(m(l)) ⊙ sign(m(s)) ⊙ sign(m(r))        (bipolar, so K⊙K = 1)
    g(w)     = (1-β)·a(w) + β · Σ_k ŵ_k · K_k ⊙ g(v_k)      (Σ ŵ = 1: contraction)
    m(w)     = Σ_k ŵ_k · K_k ⊙ g(v_k)                       (the meaning bundle)

Keys and glyphs depend on each other, and that loop has NO fixed point to
iterate to: binding scrambles every meaning when the keys change, so the
next keys are fresh — ~50% of key bits flip every round, forever (measured;
printed for --outer rounds). So glyph keys are defined in one step:
  round 0  keys from random atoms of the schema words  -> solve glyphs
  round 1  keys from those glyphs' meanings            -> solve glyphs (the result)

Two reads of the result:
  similarity  cosine of meanings m(a), m(b)
  decode      unbind one role from a word's meaning — m(w) ⊙ K(role) — and
              clean up against every glyph: does love's opposite come back as
              hate? This is the interpretive read: meaning in, the same
              structure (role -> word) out.

Run:  python open_encode2.py [--beta 0.7]
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from glyph_schema2 import ROLE_PATHS, PATHS, SCHEMA_WORDS

D = 10_000
HERE = Path(__file__).parent


def atom(name: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).choice(np.array([-1, 1], np.float32), D)


def load(path="glyphs2.jsonl", fixed=False, w_none=0.3):
    """{word: [(role_path, value, weight)]}.

    fixed=False: only the roles a word fills (glyphs differ in shape).
    fixed=True:  the FIXED lattice — every glyph binds all 27 roles; a role
                 the word leaves empty binds the word "none" (itself a glyph)
                 at weight w_none, so every glyph has the identical structure.
    """
    G = {}
    for line in open(HERE / path):
        d = json.loads(line)
        if fixed:
            G[d["word"]] = [(p, e["value"], e["weight"]) if e["value"] != "none"
                            else (p, "none", w_none) for p, e in d["glyph"].items()]
        else:
            G[d["word"]] = [(p, e["value"], e["weight"]) for p, e in d["glyph"].items()
                            if e["value"] != "none" and e["weight"] > 0]
    return {w: t for w, t in G.items() if t}


class Space:
    def __init__(self, G: dict):
        self.G = G
        self.words = sorted(G)
        self.leaves = sorted({v for t in G.values() for _, v, _ in t} - G.keys())
        self.names = self.words + self.leaves
        self.idx = {w: i for i, w in enumerate(self.names)}
        self.A = np.stack([atom(f"word:{w}") for w in self.names])

    def solve(self, K: dict, beta: float, tol=1e-4, max_iter=500):
        """Glyphs for fixed keys K {path: bipolar}. Returns (g, M, iters)."""
        T = []
        for w in self.words:
            tot = sum(wt for _, _, wt in self.G[w])
            T.append([(K[p], self.idx[v], wt / tot) for p, v, wt in self.G[w]])
        g = self.A.copy()
        for it in range(1, max_iter + 1):
            new = g.copy()
            for i, terms in enumerate(T):
                new[i] = (1 - beta) * self.A[i] + beta * sum(wt * k * g[j] for k, j, wt in terms)
            delta = float(np.abs(new - g).max())
            g = new
            if delta < tol:
                break
        M = {w: sum(wt * k * g[j] for k, j, wt in T[i]) for i, w in enumerate(self.words)}
        return g, M, it


def keys_from_atoms() -> dict:
    return {f"{l}.{s}.{r}": atom(f"word:{l}") * atom(f"word:{s}") * atom(f"word:{r}")
            for l, s, r in ROLE_PATHS}


def keys_from_meanings(M: dict) -> dict:
    sgn = {w: np.where(M[w] >= 0, 1, -1).astype(np.float32) for w in SCHEMA_WORDS}
    return {f"{l}.{s}.{r}": sgn[l] * sgn[s] * sgn[r] for l, s, r in ROLE_PATHS}


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def decode(space, g, M, word, K):
    """For each role the word fills: unbind, clean up against all glyphs."""
    G = g / np.linalg.norm(g, axis=1, keepdims=True)
    terms = sorted(space.G[word], key=lambda t: -t[2])
    probes = np.stack([M[word] * K[p] for p, _, _ in terms])     # all roles at once
    best = np.argmax(G @ probes.T, axis=0)
    return [(p.split(".")[-1], v, wt, space.names[int(b)])
            for (p, v, wt), b in zip(terms, best)]


def table(words, M):
    print("          " + "".join(f"{w:>9}" for w in words))
    for a in words:
        print(f"{a:>10}" + "".join(f"{cos(M[a], M[b]):9.2f}" for b in words))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--outer", type=int, default=3, help="key rounds to run (to show the flipping)")
    args = ap.parse_args()

    G = load()
    missing = [w for w in SCHEMA_WORDS if w not in G]
    assert not missing, f"schema words without a glyph: {missing}"
    space = Space(G)
    print(f"glyphs {len(G)}  leaves {len(space.leaves)}  schema words {len(SCHEMA_WORDS)} "
          f"(all glyphed)  β={args.beta}\n")

    K = keys_from_atoms()
    results = {}
    for rnd in range(args.outer + 1):
        g, M, iters = space.solve(K, args.beta)
        results[rnd] = (g, M, K)
        newK = keys_from_meanings(M)
        flipped = np.mean([np.mean(newK[p] != K[p]) for p in PATHS])
        label = "atom keys" if rnd == 0 else "glyph keys"
        print(f"outer {rnd} ({label}): glyphs converged in {iters} iters; "
              f"keys -> {flipped:.2%} of bits flip", flush=True)
        K = newK

    band = ["love", "hate", "kind", "mean"]
    roles = ["feeling", "mood", "agent", "patient", "before", "after", "opposite", "similar"]
    for rnd, label in ((0, "ATOM keys"), (1, "GLYPH keys")):
        g, M, K = results[rnd]
        print(f"\n=== {label} (outer {rnd}) ===")
        print("meaning similarity, band:")
        table([w for w in band if w in M], M)
        print("meaning similarity, schema words:")
        table([w for w in roles if w in M], M)
        rows = decode(space, g, M, "love", K)
        hits = sum(v == b for _, v, _, b in rows)
        print(f"decode love: {hits}/{len(rows)} roles come back exactly")
        for r, v, wt, b in rows[:10]:
            print(f"   {r:9} {v:10} ({wt})  -> {b:10} {'OK' if v == b else ''}")
        tot = ok = 0
        for w in space.words:
            for _, v, _, b in decode(space, g, M, w, K):
                tot += 1; ok += v == b
        print(f"decode ALL {len(space.words)} glyphs: {ok}/{tot} = {ok/tot:.1%} roles exact")
