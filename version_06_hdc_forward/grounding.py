"""
grounding.py — the grounding layer: one Store abstraction, two backends.
========================================================================
Dependency inversion so the SAME gate/benchmark code runs locally (here) and
against the real glyphh-runtime (in production):

  GroundingStore (interface)
    ├── LocalStore    vendored hdc.Encoder + entity records — RUNS + BENCHMARKS here
    └── RuntimeStore  targets glyphh-runtime: a DB Session + a Model row; reads the
                      Glyph store (semantic = exact symbolic; cortex = HDC bytes);
                      uses glyphh.encoder.Encoder. Lazy-imports glyphh so THIS module
                      imports fine without it; only usable where the runtime lives.

Both expose the same two structures the runtime writes on ingest:
  - EXACT symbolic store  -> authoritative grounded fillers for (entity, role)
  - HDC vector lattice    -> geometric membership verify() + similarity cohorts

Space invariant: a store advertises `space_id = sha256(vector_dim, seed, roles_config)`.
The gate MUST bind with the same space or probes return noise — `assert_space()` enforces it.

Run:  python grounding.py     # LocalStore inject -> ground -> verify (CPU, no runtime)
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

from hdc import Encoder, bind, bundle, raw_cosine


def compute_space_id(vector_dim: int, encoder_seed: int, roles_config: Optional[dict]) -> str:
    """Identical formula to glyphh-runtime/api/core/vector_space.compute_space_id."""
    normalized = json.dumps(roles_config or {}, sort_keys=True, separators=(",", ":"))
    payload = f"{vector_dim}:{encoder_seed}:{normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class GroundingStore(ABC):
    space_id: str

    @abstractmethod
    def grounded_fillers(self, entity: str, role: str, admit: float = 0.10) -> List[str]:
        """Authoritative grounded values for (entity, role) — the gate's allowed set."""

    @abstractmethod
    def verify(self, entity: str, role: str, filler: str) -> bool:
        """Geometric HDC membership check (the two structures must agree by construction)."""

    def assert_space(self, other_space_id: str) -> None:
        if self.space_id != other_space_id:
            raise ValueError(f"space_id mismatch: store={self.space_id[:8]} gate={other_space_id[:8]} "
                             "— grounding would return noise; re-encode in one space.")


# ── LocalStore: runs + benchmarks here ───────────────────────────────────────
class LocalStore(GroundingStore):
    def __init__(self, vector_dim: int = 10000, seed: int = 42, roles_config: Optional[dict] = None):
        self.enc = Encoder(vector_dim, seed)
        self.space_id = compute_space_id(vector_dim, seed, roles_config)
        self.semantic: Dict[str, Dict[str, str]] = {}     # exact symbolic store
        self.record: Dict[str, np.ndarray] = {}           # HDC cortex per entity
        self.codebook: Dict[str, Dict[str, np.ndarray]] = {}

    def inject(self, records: List[dict]) -> "LocalStore":
        """raw records -> both structures. record: {'entity':id, '<layer.role>':value,...}."""
        for rec in records:
            ent = rec["entity"]
            sem = self.semantic.setdefault(ent, {})
            for role, val in rec.items():
                if role == "entity":
                    continue
                sem[role] = str(val)
                self.codebook.setdefault(role, {})[str(val)] = self.enc.filler_vec(str(val))
        for ent, sem in self.semantic.items():
            self.record[ent] = bundle([bind(self.enc.role_vec(r), self.enc.filler_vec(v))
                                       for r, v in sem.items()])
        return self

    def probe(self, entity: str, role: str, topn: int = 3):
        if entity not in self.record:
            return []
        noisy = bind(self.record[entity], self.enc.role_vec(role))
        cands = self.codebook.get(role, {})
        return sorted(((f, raw_cosine(noisy, v)) for f, v in cands.items()), key=lambda t: -t[1])[:topn]

    def grounded_fillers(self, entity, role, admit=0.10):
        # authoritative exact value (symbolic store), cross-checked geometrically
        exact = self.semantic.get(entity, {}).get(role)
        if exact is None:
            return []
        return [exact] if self.verify(entity, role, exact) else []

    def verify(self, entity, role, filler):
        ranked = self.probe(entity, role, topn=2)
        if not ranked:
            return False
        top, s = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        return top == filler and (s - second) > 0.05


# ── RuntimeStore: targets glyphh-runtime (not runnable in this sandbox) ───────
class RuntimeStore(GroundingStore):
    """Faithful to glyphh-runtime interfaces (api/services/ingest.py, api/core/models.py).
    Construct with an open DB Session and a Model row. glyphh is imported lazily."""

    def __init__(self, db, model):
        from glyphh.encoder import Encoder as GlyphEncoder   # lazy — only where installed
        from glyphh.vector import bind as gbind
        self._db, self._model = db, model
        self._enc = GlyphEncoder(config=model.roles_config, dim=model.vector_dim, seed=model.encoder_seed)
        self._bind = gbind
        self.space_id = model.space_id

    def _glyph(self, entity: str):
        from api.core import models          # runtime package
        return self._db.get(models.Glyph, entity)   # Glyph.name is the identity key

    def grounded_fillers(self, entity, role, admit=0.10):
        g = self._glyph(entity)
        if g is None:
            return []
        val = (g.semantic or {}).get(role)            # exact symbolic value = authoritative
        if val in (None, ""):
            return []
        return [str(val)] if self.verify(entity, role, str(val)) else [str(val)]

    def verify(self, entity, role, filler):
        g = self._glyph(entity)
        if g is None:
            return False
        cortex = np.frombuffer(g.cortex, dtype=np.int8)         # HDC bytes from the store
        noisy = self._bind(cortex, self._enc.role_vec(role))    # unbind the role
        # geometric membership: does the role's slot recover this filler above the field?
        return raw_cosine(noisy, self._enc.value_vec(str(filler))) > 0.10


if __name__ == "__main__":
    store = LocalStore().inject([
        {"entity": "erin", "spatial.location": "boston", "spatial.origin": "maine",
         "relational.object": "doctor"},
        {"entity": "jack", "spatial.location": "dallas", "relational.object": "farmer"},
        {"entity": "helium", "entity.kind": "gas"},
    ])
    print(f"LocalStore space_id={store.space_id[:12]}…  entities={list(store.semantic)}")
    print("  grounded_fillers(erin, spatial.location) ->", store.grounded_fillers("erin", "spatial.location"))
    print("  grounded_fillers(helium, entity.kind)    ->", store.grounded_fillers("helium", "entity.kind"))
    print("  verify(erin, spatial.location, boston)   ->", store.verify("erin", "spatial.location", "boston"))
    print("  verify(erin, spatial.location, miami)    ->", store.verify("erin", "spatial.location", "miami"))
    # space invariant
    try:
        store.assert_space("deadbeef")
    except ValueError as e:
        print("  assert_space guard ->", str(e)[:60], "…")
    print("grounding.py OK")
