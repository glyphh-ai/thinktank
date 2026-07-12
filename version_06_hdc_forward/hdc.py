"""
hdc.py — minimal bipolar HDC layer, faithful to the glyphh-runtime semantics.
=============================================================================
The production encoder lives in the private `glyphh` package (glyphh.encoder
Encoder / glyphh.vector.bind). We can't import it here, so this vendors the
*exact same math* the runtime relies on, verified against what the runtime
services assume:

  - vectors are BIPOLAR int8 (+1 / -1), dim ~10k          (vector_space.DEFAULT_VECTOR_DIM)
  - bind(a,b)   = elementwise product                     (classic VSA role-filler bind)
  - bundle(...) = majority sign of the sum (ties -> +1)   (superposition)
  - similarity  = raw cosine = (a . b) / D  in [-1, 1]    (similarity._bipolar_raw_cosine)
  - shifted01   = (raw + 1) / 2                            (similarity.compute_pair_metrics)

Atom vectors (role_vec / value_vec) are DETERMINISTIC per (name, seed): the
runtime's Encoder(seed=...) always maps the same token to the same hypervector,
so two writes of the same fact land in the same place. We reproduce that with a
stable hash -> seeded RNG -> random ±1.

This is the "Glyph HDC lattice" side of the isomorphic pair from
adaL/docs/glyph-ada-training-loop.md.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

DEFAULT_VECTOR_DIM = 10000          # matches glyphh.config.DEFAULT_VECTOR_DIM order of magnitude
DEFAULT_ENCODER_SEED = 42           # matches vector_space.DEFAULT_ENCODER_SEED


def _stable_seed(name: str, seed: int) -> int:
    """Deterministic 63-bit seed from (name, seed) — reproducible across runs/machines."""
    h = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFFFFFFFFFF


class Encoder:
    """
    Bipolar VSA encoder. Mirrors the glyphh.encoder.Encoder surface we actually
    use in ingest.py: role_vec(role), value_vec(token), bind(), encode(fact).
    """

    def __init__(self, vector_dim: int = DEFAULT_VECTOR_DIM, seed: int = DEFAULT_ENCODER_SEED):
        self.D = int(vector_dim)
        self.seed = int(seed)

    # ---- atoms ---------------------------------------------------------------
    @lru_cache(maxsize=1 << 16)
    def _atom(self, name: str) -> tuple:
        rng = np.random.default_rng(_stable_seed(name, self.seed))
        v = rng.integers(0, 2, size=self.D, dtype=np.int8) * 2 - 1   # {0,1} -> {-1,+1}
        return tuple(v.tolist())  # cached hashable; converted to array on use

    def _atom_arr(self, name: str) -> np.ndarray:
        return np.frombuffer(np.array(self._atom(name), dtype=np.int8).tobytes(), dtype=np.int8)

    def role_vec(self, role: str) -> np.ndarray:
        return self._atom_arr(f"role::{role}")

    def value_vec(self, token: str) -> np.ndarray:
        return self._atom_arr(f"val::{str(token).lower()}")

    # ---- value of a (possibly multi-token) filler ----------------------------
    def filler_vec(self, value: str) -> np.ndarray:
        """A whole filler string -> one bipolar vector (bundle of its tokens)."""
        toks = [t for t in str(value).lower().replace("_", " ").replace("-", " ").split() if t]
        if not toks:
            return np.ones(self.D, dtype=np.int8)
        return bundle([self.value_vec(t) for t in toks])

    # ---- fact -> cortex ------------------------------------------------------
    def encode(self, role_fillers: Sequence[tuple]) -> np.ndarray:
        """
        role_fillers: list of (role, value) pairs describing one fact.
        Returns the bipolar `cortex` = bundle over bind(role, filler).
        This is `glyph.global_cortex` in the runtime.
        """
        bound = [bind(self.role_vec(r), self.filler_vec(v)) for r, v in role_fillers if v not in (None, "")]
        if not bound:
            return np.ones(self.D, dtype=np.int8)
        return bundle(bound)


# ---- primitive ops (module-level, like glyphh.vector) ------------------------
def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Role-filler bind = elementwise product of bipolar vectors. Self-inverse."""
    return (a.astype(np.int8) * b.astype(np.int8)).astype(np.int8)


def bundle(vecs: Iterable[np.ndarray]) -> np.ndarray:
    """Superpose (bundle) = majority sign of the elementwise sum; ties -> +1."""
    stack = np.stack([v.astype(np.int32) for v in vecs], axis=0)
    s = stack.sum(axis=0)
    out = np.where(s >= 0, 1, -1).astype(np.int8)
    return out


def raw_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Bipolar cosine = (a . b) / D in [-1, 1]. Identical to similarity._bipolar_raw_cosine."""
    if a.shape != b.shape:
        raise ValueError("Vector shapes must match")
    n = int(a.size)
    if n == 0:
        return 0.0
    return float((a.astype(np.int32) * b.astype(np.int32)).sum() / n)


def shifted01(a: np.ndarray, b: np.ndarray) -> float:
    return (raw_cosine(a, b) + 1.0) / 2.0


if __name__ == "__main__":
    # smoke test: the HDC algebra behaves + matches runtime expectations
    enc = Encoder(vector_dim=10000, seed=42)

    # 1. determinism: same token -> same vector across calls
    assert np.array_equal(enc.value_vec("paris"), enc.value_vec("paris"))

    # 2. quasi-orthogonality: unrelated atoms ~ 0 cosine (|sim| small at D=10k)
    s = raw_cosine(enc.value_vec("paris"), enc.value_vec("helium"))
    assert abs(s) < 0.05, s

    # 3. bind is invertible: unbind(bind(r,v), r) == v
    r, v = enc.role_vec("spatial.location"), enc.value_vec("paris")
    assert np.array_equal(bind(bind(r, v), r), v)

    # 4. a fact recovers its own filler by unbinding the role from the cortex
    fact = [("entity.name", "france"), ("spatial.location", "paris"),
            ("relational.predicate", "capital")]
    cortex = enc.encode(fact)
    #   unbind the location role, compare to the true 'paris' filler
    guess = bind(cortex, enc.role_vec("spatial.location"))
    sim_true = raw_cosine(guess, enc.filler_vec("paris"))
    sim_rand = raw_cosine(guess, enc.value_vec("helium"))
    assert sim_true > 0.15 > sim_rand, (sim_true, sim_rand)

    # 5. cortex is self-similar and unrelated facts are ~orthogonal
    other = enc.encode([("entity.name", "helium"), ("quantitative.count", "two")])
    assert raw_cosine(cortex, cortex) == 1.0
    assert abs(raw_cosine(cortex, other)) < 0.1

    print("hdc.py smoke test PASSED")
    print(f"  dim={enc.D}  bind(r,v) recovers filler: sim_true={sim_true:.3f}  sim_rand={sim_rand:.3f}")
    print(f"  cortex vs unrelated fact cosine={raw_cosine(cortex, other):+.4f} (want ~0)")
