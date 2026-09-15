"""
Strat 1 — VSA: from a judged lattice reading to the cortex vector
=================================================================
Hyperdimensional encoding of a word's lattice reading:

    role (x) value        bind    = elementwise multiply of two bipolar vecs
    segment = sign(sum_r  w_role   * (role (x) value))     weighted bundle
    layer   = sign(sum_s  w_seg    * segment)
    cortex  = sign(sum_l  w_layer  * layer)

D = 100,000 bipolar {-1,+1}. Every atom vector (role path, prime value) is
DETERMINISTIC — seeded from the sha256 of its name — so the geometry is
reproducible from the schema alone, no stored codebook needed. Roles judged
"none" or weight 0 contribute nothing. sign() keeps 0 as 0, so vectors are
ternary after bundling (v10 heritage: ternary is the house code).
"""

import hashlib

import numpy as np

from strat1_schema import LATTICE, ROLE_PATHS

D = 100_000


def atom(name: str) -> np.ndarray:
    """Deterministic bipolar vector for a named atom."""
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=D)


# Codebooks are tiny (13 roles + 65 primes) — build once at import.
ROLE_VEC = {".".join(p): atom("role:" + ".".join(p)) for p in ROLE_PATHS}

def _jitter(name: str) -> np.ndarray:
    """Deterministic per-coordinate magnitude dither in [0.5, 1.5].

    Exact weights make erasure structural: if one term's weight strictly
    exceeds the largest possible residual of the others at every coordinate,
    sign() deletes the small term everywhere (e.g. |1.0±0.9±0.8| >= 0.7 >
    0.63). Dithering each role's magnitude per coordinate turns a weight
    into the PROBABILITY that the role controls a coordinate — graded
    bundling instead of winner-take-all.
    """
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 1.5, size=D)

ROLE_JITTER = {".".join(p): _jitter("jitter:" + ".".join(p)) for p in ROLE_PATHS}
_PRIMES = sorted({v for _, seg in
                  [(k, s) for k, L in LATTICE.items() for s in L["segments"].values()]
                  for vals in seg["roles"].values() for v in vals if v != "none"})
VALUE_VEC = {v: atom("prime:" + v) for v in _PRIMES}


def encode_word(reading: dict) -> dict:
    """Judged reading -> {'segments': {...}, 'layers': {...}, 'cortex': vec}.

    The cortex is sign() of the FULL float sum — weights multiply down the
    tree (w_layer * w_seg * w_role) and every bound role reaches the cortex
    at full resolution. Ternarizing at intermediate levels erases any role
    whose weight is strictly below a sibling's (sign(1.0*a + 0.9*b) == sign(a)
    pointwise), which kills going backwards. Segment/layer vectors are
    ternary SNAPSHOTS of their own accumulators, kept for audit only.
    """
    layers = {}
    segments = {}
    cortex_acc = np.zeros(D, dtype=np.float64)
    for layer, L in LATTICE.items():
        layer_acc = np.zeros(D, dtype=np.float64)
        for seg, S in L["segments"].items():
            seg_acc = np.zeros(D, dtype=np.float64)
            for role in S["roles"]:
                path = f"{layer}.{seg}.{role}"
                entry = reading[path]
                if entry["value"] == "none" or entry["weight"] <= 0:
                    continue
                bound = ROLE_VEC[path] * VALUE_VEC[entry["value"]]
                seg_acc += entry["weight"] * ROLE_JITTER[path] * bound
            segments[f"{layer}.{seg}"] = np.sign(seg_acc).astype(np.int8)
            layer_acc += S["weight"] * seg_acc
        layers[layer] = np.sign(layer_acc).astype(np.int8)
        cortex_acc += L["weight"] * layer_acc
    cortex = np.sign(cortex_acc).astype(np.int8)
    return {"segments": segments, "layers": layers, "cortex": cortex}


def unbind(vec: np.ndarray, role_path: str) -> dict:
    """Go BACKWARDS: read a role's value out of a vector.

    Bipolar binding is self-inverse: vec * role ~ value (plus noise from the
    other bundled terms). Clean up against the prime codebook; return the
    best value and its normalized cosine so callers can threshold.
    """
    probe = vec.astype(np.float64) * ROLE_VEC[role_path]
    scores = {v: float(probe @ VALUE_VEC[v]) / D for v in VALUE_VEC}
    best = max(scores, key=scores.get)
    return {"value": best, "score": scores[best], "scores": scores}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a.astype(np.float64) @ b.astype(np.float64) / (na * nb)) if na and nb else 0.0


if __name__ == "__main__":
    # Self-test: encode a fake reading, then go backwards and demand the
    # judged values come back out of the cortex vector.
    reading = {".".join(p): {"value": "none", "weight": 0.0} for p in ROLE_PATHS}
    reading["being.substance.category"] = {"value": "someone", "weight": 1.0}
    reading["being.substance.animacy"] = {"value": "live", "weight": 0.9}
    reading["quality.value.goodness"] = {"value": "good", "weight": 0.7}
    reading["event.dynamics.agency"] = {"value": "do", "weight": 0.8}

    enc = encode_word(reading)
    print(f"D={D}  cortex nonzero: {(enc['cortex'] != 0).mean():.3f}")
    ok = True
    for path, entry in reading.items():
        if entry["value"] == "none":
            continue
        got = unbind(enc["cortex"], path)
        hit = got["value"] == entry["value"]
        ok &= hit
        print(f"  {path:<28} judged={entry['value']:<8} "
              f"decoded={got['value']:<8} score={got['score']:+.3f} "
              f"{'OK' if hit else 'MISS'}")
    print("BACKWARDS:", "PASS" if ok else "FAIL")
