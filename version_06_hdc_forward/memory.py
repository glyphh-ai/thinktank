"""
memory.py — the CORRECT groundedness primitive: entity-keyed associative memory.
================================================================================
Replaces the (wrong) whole-sentence cosine match. A statement is grounded not
because its sentence-cortex resembles a stored sentence-cortex, but because the
specific (entity, role) -> filler bindings it asserts can be RETRIEVED from the
distilled facts.

Distillation:  record[entity] = bundle over bind(role, filler) for every true
               fact about that entity.  (Kanerva / VSA associative memory.)

Query (unbind + cleanup):
    unbind(record[e], role_vec(role))  -> noisy filler vector
    cleanup against the filler codebook -> the stored filler for (e, role)

Verification of a claim "(e, role, filler)":
    membership = cos(record[e], bind(role_vec(role), filler_vec(filler)))
    grounded iff cleanup(unbind(record[e], role)) == filler  (and margin is real)

This is LOCAL and word-for-word (each role-filler is an independent probe) yet
COMPOSITIONAL (every probe is conditioned on the entity that keys the record).
Multi-hop entailment = chaining probes, with SNR decay per hop (see demo #4).

Run:  python memory.py
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from hdc import bind, bundle, raw_cosine
from corpus import build_corpus


class FactMemory:
    """Per-entity HDC records + a filler codebook for cleanup."""

    def __init__(self, enc):
        self.enc = enc
        self.records: Dict[str, np.ndarray] = {}
        self.entity_facts: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # codebook: role -> {filler -> vec}, for role-restricted cleanup
        self.codebook: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)

    def distill(self, fillers_by_fact: Dict[str, List[Tuple[str, str]]]):
        """Build entity records from the corpus's (role, filler) pairs (the vector
        side of the lattice). Entity = the fact's entity.name filler."""
        by_entity: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for rf in fillers_by_fact.values():
            ent = next((f for r, f in rf if r == "entity.name"), None)
            if ent is None:
                continue
            for role, filler in rf:
                if role == "entity.name":
                    continue                      # name is the key, not a slot to probe
                by_entity[ent].append((role, filler))
                self.codebook[role][filler] = self.enc.filler_vec(filler)
        for ent, rfs in by_entity.items():
            # dedupe identical (role,filler) so we don't over-weight repeats
            uniq = list(dict.fromkeys(rfs))
            self.entity_facts[ent] = uniq
            self.records[ent] = bundle([bind(self.enc.role_vec(r), self.enc.filler_vec(f))
                                        for r, f in uniq])
        return self

    def probe(self, entity: str, role: str, topn: int = 3):
        """unbind(record, role) -> cleanup -> ranked (filler, cosine) for this role."""
        if entity not in self.records:
            return []
        noisy = bind(self.records[entity], self.enc.role_vec(role))   # bipolar unbind == bind
        cands = self.codebook.get(role, {})
        scored = [(f, raw_cosine(noisy, v)) for f, v in cands.items()]
        scored.sort(key=lambda t: -t[1])
        return scored[:topn]

    def verify(self, entity: str, role: str, filler: str, margin: float = 0.05):
        """Is (entity, role, filler) grounded? Returns (grounded, membership, margin, top)."""
        ranked = self.probe(entity, role, topn=2)
        if not ranked:
            return False, 0.0, 0.0, None
        membership = raw_cosine(bind(self.records[entity], self.enc.role_vec(role)),
                                self.enc.filler_vec(filler))
        top_f, top_s = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        grounded = (top_f == filler) and (top_s - second > margin)
        return grounded, float(membership), float(top_s - second), top_f


if __name__ == "__main__":
    c = build_corpus(repeat=1)          # templates only — we just need the distilled fillers
    mem = FactMemory(c["enc"]).distill(c["fillers"])
    print(f"distilled {len(mem.records)} entity records "
          f"(avg {np.mean([len(v) for v in mem.entity_facts.values()]):.1f} bindings each)\n")

    print("=== 1) probe: unbind an entity by a role, clean up -> the stored filler ===")
    for ent, role in [("erin", "spatial.location"), ("erin", "spatial.origin"),
                      ("erin", "relational.object"), ("helium", "entity.kind"),
                      ("gold", "perceptual.color")]:
        ranked = mem.probe(ent, role)
        shown = ", ".join(f"{f}({s:+.3f})" for f, s in ranked)
        print(f"  probe({ent:7}, {role:20}) -> {shown}")

    print("\n=== 2) verify grounded vs hallucinated claims (the decode-time gate) ===")
    trials = [
        ("erin", "spatial.location", "boston"),   # true
        ("erin", "spatial.location", "miami"),    # false (that's frank's city)
        ("erin", "spatial.location", "austin"),   # false (carol's)
        ("helium", "entity.kind", "gas"),         # true
        ("helium", "entity.kind", "solid"),       # false
        ("gold", "perceptual.color", "gold"),     # true
        ("gold", "perceptual.color", "green"),    # false
    ]
    for ent, role, filler in trials:
        g, mship, marg, top = mem.verify(ent, role, filler)
        verdict = "GROUNDED" if g else "rejected"
        print(f"  {ent:7} {role:20} = {filler:8} -> {verdict:9} "
              f"(membership {mship:+.3f}; record returns '{top}')")

    print("\n=== 3) separation: membership of TRUE bindings vs RANDOM (entity,role,filler) ===")
    rng = np.random.default_rng(0)
    true_m, false_m = [], []
    all_fillers = list({f for rf in c["fillers"].values() for _, f in rf})
    for ent, rfs in mem.entity_facts.items():
        for role, filler in rfs:
            true_m.append(raw_cosine(bind(mem.records[ent], mem.enc.role_vec(role)),
                                     mem.enc.filler_vec(filler)))
            wrong = all_fillers[rng.integers(len(all_fillers))]
            false_m.append(raw_cosine(bind(mem.records[ent], mem.enc.role_vec(role)),
                                      mem.enc.filler_vec(wrong)))
    tm, fm = np.array(true_m), np.array(false_m)
    print(f"  true bindings : mean membership {tm.mean():+.3f}  (min {tm.min():+.3f})")
    print(f"  random claims : mean membership {fm.mean():+.3f}  (max {fm.max():+.3f})")
    print(f"  clean margin  : {tm.mean() - fm.mean():+.3f}  "
          f"(a single threshold separates them: {tm.min() > fm.max()})")

    print("\n=== 4) multi-hop = chaining probes (SNR decays per hop) ===")
    # erin lives in boston; boston lies to the east -> 'where does erin's city lie?'
    hop1 = mem.probe("erin", "spatial.location")[0][0]
    print(f"  hop1  erin . spatial.location            -> {hop1}")
    if hop1 in mem.records:
        hop2 = mem.probe(hop1, "spatial.direction")
        print(f"  hop2  {hop1} . spatial.direction  -> "
              + ", ".join(f"{f}({s:+.3f})" for f, s in hop2))
        print("  (each hop re-cleans up; depth is where HDC groundedness gets hard)")
