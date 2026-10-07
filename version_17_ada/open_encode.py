"""
OPEN ENCODE — closed-set glyphs from glyphs.jsonl (21-role open-vocab schema)
=============================================================================
Every role binds its value word's own glyph, all glyphs solved together:

    g(w) = (1-β)·a(w) + β · Σ_k ŵ_k · R_k ⊙ g(v_k)

  a(w)  identity atom of the word     R_k  role atom
  ŵ     role weights normalized so Σ ŵ = 1
A value word with no glyph yet (the closure was capped before reaching it)
is a leaf: g(v) = a(v). Binding is norm-preserving and Σ ŵ = 1, so the map
is a contraction with factor β: iteration from g = a always converges.
(Normalizing Σ ŵ² = 1 instead diverges: binding is self-inverse, so
symmetric pairs — love -opposite-> hate -opposite-> love — feed a glyph
back into itself undistorted, and those loops grow without bound.)

Words are compared by their MEANING m(w) = Σ ŵ R⊙g(v), not the full glyph:
~15 near-orthogonal terms with Σ ŵ = 1 have norm ~1/√15, so in g the
identity atom swamps the meaning; cosine of m is scale-free.


Run:  python open_encode.py [--beta 0.7] [words ...]
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from glyph_schema import PATHS

D = 10_000
HERE = Path(__file__).parent


def atom(name: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).choice(np.array([-1, 1], np.float32), D)


def load():
    G = {}
    for line in open(HERE / "glyphs.jsonl"):
        d = json.loads(line)
        G[d["word"]] = [(p, e["value"], e["weight"]) for p, e in d["glyph"].items()
                        if e["value"] != "none" and e["weight"] > 0]
    return G


def solve(G: dict, beta: float, tol=1e-5, max_iter=500):
    words = sorted(G)
    leaves = sorted({v for t in G.values() for _, v, _ in t} - G.keys())
    idx = {w: i for i, w in enumerate(words + leaves)}
    A = np.stack([atom(f"word:{w}") for w in words + leaves])
    ROLE = {p: atom(f"role:{p}") for p in PATHS}
    T = []
    for w in words:
        tot = sum(wt for _, _, wt in G[w])
        T.append([(ROLE[p], idx[v], wt / tot) for p, v, wt in G[w]])
    g = A.copy()
    for it in range(1, max_iter + 1):
        new = g.copy()
        for i, terms in enumerate(T):
            new[i] = (1 - beta) * A[i] + beta * sum(wt * R * g[j] for R, j, wt in terms)
        delta = float(np.abs(new - g).max())
        g = new
        if delta < tol:
            M = {w: sum(wt * R * g[j] for R, j, wt in T[i]) for i, w in enumerate(words)}
            return M, it, len(leaves)
    raise RuntimeError(f"no convergence (delta {delta:.2e})")


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("words", nargs="*",
                    default=["love", "hate", "kind", "mean", "care", "anger", "good", "bad"])
    args = ap.parse_args()
    G = load()
    M, iters, n_leaves = solve(G, args.beta)
    print(f"glyphs {len(G)}  leaves {n_leaves}  β={args.beta}  converged in {iters} iterations\n")
    ws = [w for w in args.words if w in M]
    print("        " + "".join(f"{w:>8}" for w in ws))
    for a in ws:
        print(f"{a:>8}" + "".join(f"{cos(M[a], M[b]):8.2f}" for b in ws))
