"""
TYPED ENCODE — typed atomic construction (see DESIGN.md)
========================================================
Same v3 glyphs (27 real words per word), assembled by the schema's anatomy:

  roles     typed association — attach / inherit / attract / negate /
            contain / order / direct (ROLE_TYPE below)
  segments  typed composition — bundle / together (frame) / sequence
  layers    stacked, each a keyed vector you can peel off the cortex

Two spaces, both solved as contractions (always converge):

  structure  S(w) = Σ_roles  K_path ⊙ ρ^k(g(v))  [+ frame product]
             g(w) = (1-β)·a(w) + β·S(w)              the stored, decodable glyph
             (S is NOT rescaled: each term weighs 1/27, so the map is a contraction;
              rescaling lets the hub component every glyph shares grow each round
              until all glyphs collapse onto it — measured: decode 10.6%, cos 0.99)
  meaning    μ(w) = Ŝ(w) + γ · Σ_pull s_k · μ(v_k) / n_pull
             s = +1 inherit/attract, −1 negate        where the word SITS

K_path = sign(m(layer)) ⊙ sign(m(segment)) ⊙ sign(m(role)) — keys are the schema
words' own glyphs. ρ = cyclic shift (exact inverse). Untyped baseline: every
role 'attach', no pulls, no frame product — today's v3 encoding.

Run:  python typed_encode.py
"""

import json
from pathlib import Path

import numpy as np

import open_encode2 as E
from glyph_schema3 import SCHEMA, ROLE_PATHS, PATHS, SCHEMA_WORDS

HERE = Path(__file__).parent
D = E.D

# ---- the anatomy -------------------------------------------------------------
ROLE_TYPE = {
    "class": "attach", "number": "attach", "tense": "attach", "degree": "attach",
    "slot": "attach", "register": "attach", "mood": "attach", "feeling": "attach",
    "goodness": "attach", "strength": "attach", "place": "attach", "time": "attach",
    "base": "inherit", "category": "inherit",
    "similar": "attract", "partner": "attract", "figure": "attract", "phrase": "attract",
    "opposite": "negate",
    "part": "contain", "whole": "contain",
    "before": "order", "after": "order",
    "cause": "direct", "purpose": "direct", "agent": "direct", "patient": "direct",
}
SHIFT = {"part": 1, "whole": -1, "before": -2, "after": 2,       # ρ^k per role
         "cause": -3, "purpose": 3, "agent": -4, "patient": 4}
PULL = {"inherit": +1, "attract": +1, "negate": -1}
SEGMENT_TYPE = {"frame": "together", "position": "sequence"}     # others: bundle
assert set(ROLE_TYPE) == {r for _, _, r in ROLE_PATHS}


def rho(x, k):
    return np.roll(x, k) if k else x


class Typed:
    def __init__(self, typed=True, beta=0.7, gamma=0.5, glyphs="glyphs3.jsonl"):
        self.typed, self.beta, self.gamma = typed, beta, gamma
        rows = [json.loads(l) for l in open(HERE / glyphs)]
        self.G = {d["word"]: {p: e["value"] for p, e in d["glyph"].items()} for d in rows}
        self.words = sorted(self.G)
        leaves = sorted({v for g in self.G.values() for v in g.values()} - self.G.keys())
        self.names = self.words + leaves
        self.idx = {w: i for i, w in enumerate(self.names)}
        self.A = np.stack([E.atom(f"word:{w}") for w in self.names])
        self.K = self._glyph_keys()

    # keys from the schema words' own glyphs (one step from atom keys; see open_encode2)
    def _glyph_keys(self):
        G2 = {w: [(p, v, 1.0) for p, v in g.items()] for w, g in self.G.items()}
        space = E.Space(G2)
        _, M0, _ = space.solve(E.keys_from_atoms(), self.beta)
        return E.keys_from_meanings(M0)

    def shift(self, role):
        return SHIFT.get(role, 0) if self.typed else 0

    def structure(self, w, g):
        terms = []
        for (l, s, r) in ROLE_PATHS:
            p = f"{l}.{s}.{r}"
            terms.append(self.K[p] * rho(g[self.idx[self.G[w][p]]], self.shift(r)))
        S = sum(terms) / len(terms)
        if self.typed:                      # frame is 'together': the event as one term
            f = self.G[w]
            ev = (self.K["sentence.frame.agent"] * g[self.idx[f["sentence.frame.agent"]]]
                  * rho(g[self.idx[f["sentence.frame.patient"]]], 1))
            S = S + ev / len(terms)
        return S

    def solve(self, tol=1e-5, max_iter=300):
        n = len(self.words)
        g = self.A.copy()
        for _ in range(max_iter):                       # structure: g = (1-β)a + β·S
            S = np.stack([self.structure(w, g) for w in self.words])
            new = g.copy()
            new[:n] = (1 - self.beta) * self.A[:n] + self.beta * S
            delta = float(np.abs(new - g).max()); g = new
            if delta < tol:
                break
        self.g, self.S = g, S
        Shat = S / np.linalg.norm(S, axis=1, keepdims=True)
        mu = Shat.copy()                                # meaning: μ = Ŝ + γ Σ s μ(v)/n
        pulls = []
        for w in self.words:
            pl = [(PULL[ROLE_TYPE[r]], self.idx[self.G[w][f"{l}.{s}.{r}"]])
                  for l, s, r in ROLE_PATHS if self.typed and ROLE_TYPE[r] in PULL]
            pulls.append([(sg, j) for sg, j in pl if j < n])   # leaves have no meaning
        for _ in range(max_iter):
            new = Shat.copy()
            for i, pl in enumerate(pulls):
                if pl:
                    new[i] += self.gamma * sum(sg * mu[j] for sg, j in pl) / len(pl)
            delta = float(np.abs(new - mu).max()); mu = new
            if delta < tol:
                break
        self.mu = {w: mu[i] for i, w in enumerate(self.words)}
        return self

    # ---- reads -----------------------------------------------------------
    def decode(self, w):
        mem = self.g / np.linalg.norm(self.g, axis=1, keepdims=True)
        S = self.S[self.idx[w]]
        probes = np.stack([rho(S * self.K[f"{l}.{s}.{r}"], -self.shift(r))
                           for l, s, r in ROLE_PATHS])
        best = np.argmax(mem @ probes.T, axis=0)
        return {p: self.names[int(b)] for p, b in zip(PATHS, best)}

    def layer(self, w, layer):
        """Peel one layer: the role terms of that layer, read back out of the cortex."""
        return np.concatenate([self.S[self.idx[w]] * self.K[p] for p in PATHS
                               if p.startswith(layer + ".")])

    def cos(self, a, b):
        x, y = self.mu[a], self.mu[b]
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def report(T, label, band):
    print(f"\n=== {label} ===")
    print("meaning μ similarity:")
    print("          " + "".join(f"{w:>8}" for w in band))
    for a in band:
        print(f"{a:>10}" + "".join(f"{T.cos(a, b):8.2f}" for b in band))
    tot = ok = swapped = order_n = pw = pw_n = 0
    for w in T.words:
        d, g = T.decode(w), T.G[w]
        for p in PATHS:
            tot += 1; ok += d[p] == g[p]
        b, a = "sentence.position.before", "sentence.position.after"
        if g[b] != g[a]:
            order_n += 1; swapped += d[b] == g[a]
        pt, wh = "meaning.relation.part", "meaning.relation.whole"
        if g[pt] != g[wh]:
            pw_n += 1; pw += d[pt] == g[wh]
    print(f"decode: {ok}/{tot} = {ok / tot:.1%} roles exact")
    print(f"direction: 'before' decoded as the after-word {swapped}/{order_n}; "
          f"'part' decoded as the whole-word {pw}/{pw_n}")
    return ok / tot


if __name__ == "__main__":
    band = ["love", "hate", "kind", "mean", "care", "nice", "anger", "joy"]
    U = Typed(typed=False).solve()
    band = [w for w in band if w in U.mu]
    acc_u = report(U, "UNTYPED (every role attach — today's v3)", band)
    T = Typed(typed=True).solve()
    acc_t = report(T, "TYPED (DESIGN.md anatomy)", band)

    print("\n=== pre-registered tests ===")
    def verdict(ok): return "PASS" if ok else "FAIL"
    for a, b in (("love", "hate"), ("kind", "mean")):
        print(f"1 opposites point away   {a}/{b}: untyped {U.cos(a, b):+.2f} -> typed "
              f"{T.cos(a, b):+.2f}   {verdict(T.cos(a, b) < 0)}")
    for a, b in (("love", "care"), ("kind", "nice")):
        if a in T.mu and b in T.mu:
            print(f"2 synonyms stay close    {a}/{b}: untyped {U.cos(a, b):+.2f} -> typed "
                  f"{T.cos(a, b):+.2f}   {verdict(T.cos(a, b) > 0)}")
    print(f"3 structure survives     decode {acc_u:.1%} -> {acc_t:.1%}   {verdict(acc_t >= 0.90)}")
