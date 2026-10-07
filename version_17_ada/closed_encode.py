"""
CLOSED ENCODE — every value is a glyph, not an atom
===================================================
v13 bound each role to a random prime ATOM: role ⊙ atom(good). Here the set
is closed — every prime has its own judged reading (53 from readings.jsonl,
12 multi-word primes from prime_readings.jsonl) — so a role binds the value's
own GLYPH: role ⊙ g(good). Glyphs are defined by each other (and primes by
themselves: good = goodness: good), so they are solved together:

    g(w) = a(w) + β · Σ_k ŵ_k · R_k ⊙ g(v_k)

  a(w)  the word's identity atom ("it is a word")
  R_k   role vector, v_k the role's value, ŵ_k the lattice weight
        (layer × segment × role, normalized to sum 1 over the word's roles)
  β<1   how much of a glyph is its meaning vs its identity

Binding is ±1 elementwise (norm-preserving) and ŵ sums to 1, so the map is a
contraction with factor β: iterating from g = a converges to the unique
fixed point, error ≤ β^t. Only the 65 primes appear as values, so the fixed
point is solved on the primes; every other word is then one pass.

Run:  python closed_encode.py            # solve + compare words, v13 vs closed
"""

import json
from pathlib import Path

import numpy as np

from strat1_schema import LATTICE, ALL_PRIMES
from strat1_vsa import D, atom, ROLE_VEC, cosine, encode_word

HERE = Path(__file__).parent


def load_readings() -> dict:
    R = {}
    for name in ("readings.jsonl", "prime_readings.jsonl"):
        for line in open(HERE / name):
            d = json.loads(line)
            R[d["word"]] = d["reading"]
    return R


def terms(reading: dict) -> list:
    """[(role_path, value, ŵ)] — weights multiplied down the lattice, summed to 1."""
    out = []
    for layer, L in LATTICE.items():
        for seg, S in L["segments"].items():
            for role in S["roles"]:
                e = reading[f"{layer}.{seg}.{role}"]
                if e["value"] != "none" and e["weight"] > 0:
                    out.append((f"{layer}.{seg}.{role}", e["value"],
                                L["weight"] * S["weight"] * e["weight"]))
    total = sum(w for _, _, w in out)
    return [(p, v, w / total) for p, v, w in out]


def check_closed(R: dict):
    values = {v for r in R.values() for _, v, _ in terms(r)}
    missing = sorted(values - R.keys())
    assert not missing, f"not closed — no glyph for {missing}"
    return values


def solve_primes(R: dict, beta: float, tol=1e-6, max_iter=500):
    """Fixed point g = a + β Σ ŵ R⊙g(v) over the primes. Returns ({prime: g}, iters)."""
    T = {p: terms(R[p]) for p in ALL_PRIMES}
    A = {p: atom(f"word:{p}").astype(np.float64) for p in ALL_PRIMES}
    g = dict(A)
    for it in range(1, max_iter + 1):
        new = {p: A[p] + beta * sum(w * ROLE_VEC[path] * g[v] for path, v, w in T[p])
               if T[p] else A[p] for p in ALL_PRIMES}
        delta = max(np.abs(new[p] - g[p]).max() for p in ALL_PRIMES)
        g = new
        if delta < tol:
            return g, it
    raise RuntimeError(f"no convergence after {max_iter} iterations (delta {delta:.2e})")


def glyph(word: str, R: dict, G: dict, beta: float) -> np.ndarray:
    """A word's closed glyph (float), from the solved prime glyphs."""
    if word in G:
        return G[word]
    return atom(f"word:{word}").astype(np.float64) + beta * sum(
        w * ROLE_VEC[path] * G[v] for path, v, w in terms(R[word]))


def table(words, vec):
    print("        " + "".join(f"{w:>8}" for w in words))
    for a in words:
        print(f"{a:>8}" + "".join(f"{cosine(vec[a], vec[b]):8.2f}" for b in words))


if __name__ == "__main__":
    R = load_readings()
    values = check_closed(R)
    print(f"glyphs: {len(R)}  value words: {len(values)}  -> CLOSED "
          f"(every value word has its own glyph)\n")

    words = ["love", "hate", "kind", "mean", "gentle", "nice", "dog", "table"]

    print("v13 — values are random atoms (sign of cortex):")
    table(words, {w: encode_word(R[w])["cortex"] for w in words})

    for beta in (0.5, 0.9):
        G, iters = solve_primes(R, beta)
        print(f"\nclosed — values are glyphs, β={beta} (converged in {iters} iterations), sign(g):")
        table(words, {w: np.sign(glyph(w, R, G, beta)) for w in words})
