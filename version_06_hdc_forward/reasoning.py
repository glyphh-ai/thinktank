"""
reasoning.py — compositional reasoning over the HIERARCHICAL HDC lattice.
=========================================================================
memory.py encodes each entity as a FLAT bundle: record = Σ bind(role, filler).
That gives clean single-fact membership but treats multi-hop as repeated flat
lookups (noise compounds every hop) and cannot address a subtree.

Ada's lattice is a TREE (cortex -> layer -> role -> filler). We encode that tree
with NESTED binds:

    layer_node[L] = Σ_{role in L}  bind(role_key(role), filler_vec(filler))
    cortex        = Σ_{layer L}    bind(layer_key(L),   layer_node[L])

This unlocks three things a flat bundle cannot do:

  1. SUBTREE EXTRACTION   unbind(cortex, layer_key) -> all of a layer's facts.
  2. BOUNDED MULTI-HOP    cleanup at each tier boundary restores SNR, so k-hop
                          chains don't accumulate noise linearly.
  3. RELATIONAL ANALOGY   bind(rec_B, bind(rec_A, filler_A)) transfers a relation
                          from A to B with no role ever named (VSA analogy).

Run:  python reasoning.py
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from hdc import Encoder, bind, bundle, raw_cosine
from corpus import build_corpus, UNIVERSAL_SCHEMA
from memory import FactMemory

LAYERS = list(UNIVERSAL_SCHEMA.keys())   # entity, perceptual, spatial, temporal, ...


class HierMemory:
    """Hierarchical (tree) HDC records: cortex -> layer -> role -> filler."""

    def __init__(self, enc: Encoder):
        self.enc = enc
        self.cortex: Dict[str, np.ndarray] = {}
        self.facts: Dict[str, List[Tuple[str, str]]] = {}
        self.codebook: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)  # role -> filler -> vec
        self.concept: Dict[str, np.ndarray] = {}   # clean vector for any entity/filler name

    def layer_key(self, layer: str) -> np.ndarray:
        return self.enc.role_vec(f"layer::{layer}")

    def distill(self, fillers_by_fact):
        by_entity: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for rf in fillers_by_fact.values():
            ent = next((f for r, f in rf if r == "entity.name"), None)
            if ent is None:
                continue
            for role, filler in rf:
                if role == "entity.name":
                    continue
                by_entity[ent].append((role, filler))
                self.codebook[role][filler] = self.enc.filler_vec(filler)
                self.concept[filler] = self.enc.filler_vec(filler)
        for ent, rfs in by_entity.items():
            uniq = list(dict.fromkeys(rfs))
            self.facts[ent] = uniq
            self.concept[ent] = self.enc.filler_vec(ent)
            self.cortex[ent] = self._build_tree(uniq)
        return self

    def _build_tree(self, rfs: List[Tuple[str, str]]) -> np.ndarray:
        by_layer: Dict[str, list] = defaultdict(list)
        for role, filler in rfs:
            by_layer[role.split(".")[0]].append(
                bind(self.enc.role_vec(role), self.enc.filler_vec(filler)))
        layer_nodes = []
        for layer, leaves in by_layer.items():
            layer_nodes.append(bind(self.layer_key(layer), bundle(leaves)))
        return bundle(layer_nodes)

    # ---- 1) hierarchical descent: cortex -> layer -> role -> filler ----------
    def get(self, entity: str, role: str, topn: int = 2):
        if entity not in self.cortex:
            return []
        layer = role.split(".")[0]
        sub = bind(self.cortex[entity], self.layer_key(layer))    # unbind the layer
        leaf = bind(sub, self.enc.role_vec(role))                 # unbind the role
        cands = self.codebook.get(role, {})
        scored = sorted(((f, raw_cosine(leaf, v)) for f, v in cands.items()), key=lambda t: -t[1])
        return scored[:topn]

    # ---- 2) subtree extraction: everything in a layer -----------------------
    def layer_query(self, entity: str, layer: str, per_role_top: int = 1):
        """Extract a whole layer sub-bundle and read out each role present."""
        if entity not in self.cortex:
            return {}
        out = {}
        for role in [f"{layer}.{r}" for r in UNIVERSAL_SCHEMA[layer]]:
            hits = self.get(entity, role, topn=1)
            if hits and hits[0][1] > 0.08:      # role populated for this entity
                out[role] = hits[0]
        return out

    # ---- 3) bounded multi-hop with cleanup at each node ---------------------
    def chain(self, start: str, path: List[str], cleanup: bool = True, verbose=False):
        """Follow role1 -> role2 -> ... across records, cleaning to a clean concept
        at each hop (the hierarchy's tier boundary). Returns (value, margin_trace)."""
        cur = start
        trace = []
        for role in path:
            hits = self.get(cur, role, topn=2)
            if not hits:
                trace.append((role, None, 0.0)); return None, trace
            top, s = hits[0]
            margin = s - (hits[1][1] if len(hits) > 1 else 0.0)
            trace.append((role, top, margin))
            cur = top if cleanup else top        # cleanup => next hop keys off the clean concept name
            if verbose:
                print(f"      {role:22} -> {top:10} (margin {margin:+.3f})")
        return cur, trace

    # ---- 4) relational analogy: transfer a relation A->B (role DISCOVERED) ----
    def analogy(self, a: str, filler_in_a: str, b: str, layer: str, topn: int = 2):
        """'filler_in_a is to a as WHAT is to b' — DISCOVER the role that binds
        filler_in_a in a's tree (cleaned to a known role key), then apply it to b.
        The cleanup is essential: the raw extracted role carries a filler residue
        that would otherwise leak straight back (see the maine-leak bug)."""
        sub_a = bind(self.cortex[a], self.layer_key(layer))
        role_noisy = bind(sub_a, self.enc.filler_vec(filler_in_a))
        # cleanup the extracted role to the nearest clean role key in this layer
        roles = [f"{layer}.{r}" for r in UNIVERSAL_SCHEMA[layer]]
        role = max(roles, key=lambda r: raw_cosine(role_noisy, self.enc.role_vec(r)))
        sub_b = bind(self.cortex[b], self.layer_key(layer))
        leaf_b = bind(sub_b, self.enc.role_vec(role))              # apply the CLEAN key
        cands = self.codebook.get(role, {})
        ranked = sorted(((f, raw_cosine(leaf_b, v)) for f, v in cands.items()), key=lambda t: -t[1])
        return role, ranked[:topn]


def depth_demo(enc, D_hops=6, fanout=4, seed=0):
    """DEEPER reasoning = retrieve -> cleanup -> retrieve, with the hierarchy's tiers
    as the cleanup boundaries. Build a chain n0 -> n1 -> ... and a distractor fan-out
    per node, then walk it. Cleanup at each hop resets noise, so depth is unbounded
    as long as each single hop clears the SNR bar (set by fan-out / capacity)."""
    rng = np.random.default_rng(seed)
    names = [f"n{i}" for i in range(D_hops + 1)]
    codebook = {nm: enc.filler_vec(nm) for nm in names}
    distract = [f"d{i}" for i in range(200)]
    for d in distract:
        codebook[d] = enc.filler_vec(d)
    rec = {}
    for i, nm in enumerate(names):
        leaves = []
        if i < D_hops:
            leaves.append(bind(enc.role_vec("relational.object"), enc.filler_vec(names[i + 1])))
        # fan-out distractors in other roles (load the record)
        for _ in range(fanout):
            leaves.append(bind(enc.role_vec("perceptual.color"),
                               enc.filler_vec(distract[rng.integers(len(distract))])))
        rec[nm] = bundle(leaves)

    def hop(vec_or_name, cleanup):
        rc = rec[vec_or_name] if cleanup else vec_or_name
        leaf = bind(rc, enc.role_vec("relational.object"))
        scored = sorted(((nm, raw_cosine(leaf, codebook[nm])) for nm in names), key=lambda t: -t[1])
        return scored[0]

    print("  walk n0 -> ... with cleanup at each tier boundary:")
    cur, ok = "n0", True
    for step in range(D_hops):
        nxt, margin = hop(cur, cleanup=True)
        exp = names[names.index(cur) + 1]
        good = nxt == exp
        ok &= good
        print(f"    hop {step+1}: {cur} -> {nxt:4} (margin {margin:+.3f}) {'ok' if good else 'WRONG'}")
        cur = nxt
    print(f"  reached {cur} in {D_hops} hops — {'all correct' if ok else 'derailed'} "
          f"(each hop's noise reset by cleanup; depth is not the limiter, per-hop SNR is)")


def capacity_test(enc, seed=0):
    """Leaf-retrieval margin as facts-per-entity grows: FLAT vs HIERARCHICAL.
    Facts spread across Ada's 7 layers; hierarchy prunes to a small subtree per query."""
    rng = np.random.default_rng(seed)
    roles = [f"{L}.{r}" for L, rs in UNIVERSAL_SCHEMA.items() for r in rs]
    fillers = [f"f{i}" for i in range(400)]
    print(f"  {'facts/entity':>12} {'FLAT margin':>12} {'HIER margin':>12}   winner")
    for k in (4, 8, 16, 24, 32):
        flat_m, hier_m = [], []
        for _ in range(40):
            chosen_roles = list(rng.choice(len(roles), size=min(k, len(roles)), replace=False))
            rf = [(roles[i], fillers[rng.integers(len(fillers))]) for i in chosen_roles]
            # flat
            flat = bundle([bind(enc.role_vec(r), enc.filler_vec(f)) for r, f in rf])
            # hierarchical
            by_layer = defaultdict(list)
            for r, f in rf:
                by_layer[r.split(".")[0]].append(bind(enc.role_vec(r), enc.filler_vec(f)))
            hier = bundle([bind(enc.role_vec(f"layer::{L}"), bundle(v)) for L, v in by_layer.items()])
            # query one random stored (role, filler); margin = cos(true) - cos(best wrong)
            qr, qf = rf[rng.integers(len(rf))]
            wrongs = [enc.filler_vec(fillers[rng.integers(len(fillers))]) for _ in range(20)]
            fl_leaf = bind(flat, enc.role_vec(qr))
            flat_m.append(raw_cosine(fl_leaf, enc.filler_vec(qf)) - max(raw_cosine(fl_leaf, w) for w in wrongs))
            hi_leaf = bind(bind(hier, enc.role_vec(f"layer::{qr.split('.')[0]}")), enc.role_vec(qr))
            hier_m.append(raw_cosine(hi_leaf, enc.filler_vec(qf)) - max(raw_cosine(hi_leaf, w) for w in wrongs))
        fm, hm = float(np.mean(flat_m)), float(np.mean(hier_m))
        print(f"  {k:>12} {fm:>12.3f} {hm:>12.3f}   {'HIER' if hm > fm else 'flat'}")


if __name__ == "__main__":
    c = build_corpus(repeat=1)
    enc = c["enc"]
    hm = HierMemory(enc).distill(c["fillers"])
    flat = FactMemory(enc).distill(c["fillers"])
    print(f"distilled {len(hm.cortex)} hierarchical entity trees\n")

    print("=== 1) hierarchical descent (cortex -> layer -> role -> filler) ===")
    for ent, role in [("erin", "spatial.location"), ("erin", "spatial.origin"),
                      ("gold", "perceptual.color"), ("helium", "entity.kind")]:
        h = hm.get(ent, role)
        print(f"  get({ent:6}, {role:20}) -> " + ", ".join(f"{f}({s:+.3f})" for f, s in h))

    print("\n=== 2) subtree extraction — all of an entity's SPATIAL facts at once ===")
    for ent in ("erin", "boston"):
        sub = hm.layer_query(ent, "spatial")
        print(f"  {ent:7} spatial layer -> " + ", ".join(f"{r.split('.')[1]}={f}" for r, (f, s) in sub.items()))
    print("  (flat memory cannot address a layer — this needs the tree)")

    print("\n=== 3) bounded multi-hop: 'what direction is the city erin lives in?' ===")
    val, trace = hm.chain("erin", ["spatial.location", "spatial.direction"], verbose=True)
    print(f"  answer: {val}   (erin -> boston -> east; cleanup to a clean concept between hops)")

    print("\n=== 4) relational analogy — 'maine is to erin as WHAT is to jack?' ===")
    role, ana = hm.analogy("erin", "maine", "jack", "spatial")
    print(f"  discovered role: {role}")
    print(f"  origin(erin)=maine  ->  {role}(jack) = " + ", ".join(f"{f}({s:+.3f})" for f, s in ana))
    print(f"  (ground truth: jack is from kansas)")

    print("\n=== 5) DEEPER reasoning — bounded-error multi-hop via tiered cleanup ===")
    depth_demo(enc)

    print("\n=== 6) capacity (honest) — leaf-retrieval margin vs facts/entity ===")
    print("  within capacity, a single flat bundle beats double-sign-bundled hierarchy")
    print("  on raw leaf SNR; hierarchy's payoff is STRUCTURE (subtree addressing,")
    print("  cleanup boundaries for depth), not leaf SNR:")
    capacity_test(enc)
