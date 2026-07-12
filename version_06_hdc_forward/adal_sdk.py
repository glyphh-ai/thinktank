"""
adal_sdk.py — the bolt-in SDK: raw data -> two structures -> grounded any-LLM gate.
=================================================================================
One facade over everything in version_06. Three surfaces:

  WRITE   ingest(records)            raw records -> BOTH structures on one write path:
                                       - exact STORE  (entity -> {role: value})   [aggregation]
                                       - HDC LATTICE  (entity-keyed bipolar records) [grounding]
  READ    probe / verify / grounded_fillers / aggregate     query either structure
  GATE    constraint(tokenizer, prompt, entity, role)       a logits processor ANY
                                       logit-access LLM bolts into its decode loop.

The GATE is the "bolt into any LLM" seam. Integration matrix (honest):
  - logit access (mlx-lm / HF transformers / vLLM / llama.cpp) -> HARD guarantee:
    the model cannot emit a non-Ada filler (mask before commit). `constraint()`
    returns a callable(tokens, logits)->logits; wrap it per runtime's hook.
  - API models (OpenAI/Anthropic, no logit access) -> verify-then-refuse: generate,
    parse asserted (entity, role, filler)s, verify() each, refuse/repair ungrounded.
    Weaker (post-hoc) but still enforces groundedness.

This wraps the SAME encoder the production runtime uses (glyphh.encoder.Encoder;
here the vendored hdc.Encoder). In product, point it at glyphh-runtime's store.

Run:  python adal_sdk.py        # ingest -> probe -> verify -> gate (CPU, no model)
"""
from __future__ import annotations

from typing import Dict, List, Optional

from hdc import Encoder
from memory import FactMemory
from lattice_adapter import GroundedConstraint, build_allowed


class AdaL:
    def __init__(self, vector_dim: int = 10000, seed: int = 42):
        self.enc = Encoder(vector_dim, seed)
        self.store: Dict[str, Dict[str, object]] = {}     # exact symbolic side
        self._fillers: Dict[str, list] = {}               # fact_key -> [(role, filler)]
        self.mem: Optional[FactMemory] = None             # HDC vector side

    # ── WRITE: raw records -> both structures ────────────────────────────────
    def ingest(self, records: List[dict]) -> "AdaL":
        """records: dicts with an 'entity' key + Ada role:value pairs, e.g.
        {'entity':'erin', 'spatial.location':'boston', 'spatial.origin':'maine'}."""
        for i, rec in enumerate(records):
            ent = rec["entity"]
            self.store.setdefault(ent, {})
            rf = [("entity.name", ent)]
            for role, val in rec.items():
                if role == "entity":
                    continue
                self.store[ent][role] = val               # exact store
                rf.append((role, str(val)))               # HDC (categorical/id fillers)
            self._fillers[f"f{i}"] = rf
        self.mem = FactMemory(self.enc).distill(self._fillers)   # rebuild vector side
        return self

    # ── READ ────────────────────────────────────────────────────────────────
    def probe(self, entity: str, role: str, topn: int = 3):
        return self.mem.probe(entity, role, topn)

    def verify(self, entity: str, role: str, filler: str):
        """(grounded, membership, margin, top) — the API-model gate."""
        return self.mem.verify(entity, role, filler)

    def grounded_fillers(self, entity: str, role: str, admit: float = 0.10) -> List[str]:
        return [f for f, s in self.probe(entity, role, topn=6) if s >= admit]

    def aggregate(self, role: str, agg: str = "sum", where: Optional[dict] = None):
        """Exact reducer over the STORE (never over vectors)."""
        vals = [s[role] for e, s in self.store.items()
                if role in s and all(s.get(k) == v for k, v in (where or {}).items())]
        nums = [float(v) for v in vals if isinstance(v, (int, float))]
        if agg == "count":
            return len(vals)
        if not nums:
            return 0
        return sum(nums) if agg == "sum" else sum(nums) / len(nums)

    # ── GATE: the bolt-in logits processor ───────────────────────────────────
    def constraint(self, tokenizer, prompt: str, entity: str, role: str,
                   refusal: str = "unknown", eos_id: int = 0) -> GroundedConstraint:
        fillers = self.grounded_fillers(entity, role)
        seqs, _ = build_allowed(tokenizer, prompt, fillers, refusal)
        return GroundedConstraint(seqs, eos_id)


# ── tiny stub tokenizer for a model-free demo ────────────────────────────────
class _StubTok:
    def __init__(self, words): self.stoi = {w: i for i, w in enumerate(words)}
    def encode(self, s): return [self.stoi[w] for w in s.split()]


if __name__ == "__main__":
    import numpy as np

    ada = AdaL().ingest([
        {"entity": "erin", "spatial.location": "boston", "spatial.origin": "maine",
         "relational.object": "doctor", "quantitative.magnitude": 120000},
        {"entity": "jack", "spatial.location": "dallas", "relational.object": "farmer",
         "quantitative.magnitude": 85000},
        {"entity": "helium", "entity.kind": "gas"},
    ])
    print("WRITE  ingested", len(ada.store), "entities -> exact store + HDC lattice")

    print("\nREAD")
    print("  probe(erin, spatial.location) ->", ada.probe("erin", "spatial.location", topn=2))
    print("  verify(erin, spatial.location, boston) ->", ada.verify("erin", "spatial.location", "boston")[:1],
          "| miami ->", ada.verify("erin", "spatial.location", "miami")[:1])
    print("  aggregate(quantitative.magnitude, sum) -> $", int(ada.aggregate("quantitative.magnitude", "sum")))

    print("\nGATE (bolt-in) — allowed next tokens are Ada's grounded fillers or refusal")
    vocab = ["Erin", "lives", "in", "boston", "miami", "dallas", "<eos>", "unknown"]
    tok = _StubTok(vocab)
    con = ada.constraint(tok, "Erin lives in", "erin", "spatial.location",
                         refusal="unknown", eos_id=vocab.index("<eos>"))
    logits = np.zeros(len(vocab)); logits[vocab.index("miami")] = 9.0   # model WANTS miami
    masked = con(np.array(tok.encode("Erin lives in")), logits)
    print("  model biased to 'miami' but gate forces ->", vocab[int(np.argmax(masked))])
    assert vocab[int(np.argmax(masked))] == "boston"

    print("\n3-line bolt-in for a real local LLM (mlx-lm / HF / vLLM):")
    print("  ada = AdaL().ingest(rows)")
    print("  con = ada.constraint(tok, prompt, entity, role, eos_id=tok.eos_token_id)")
    print("  generate(model, tok, prompt, logits_processors=[con])   # cannot answer Ada wrong")
