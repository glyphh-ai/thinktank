"""
run_qwen.py — Ada in-loop memory adapter on a STOCK Qwen3 (MLX, Apple Silicon).
==============================================================================
Proves the killer property on a real open model: whatever Qwen3 was trained on,
it cannot answer an Ada-typed fact incorrectly — the in-loop lattice adapter masks
its logits, at every step, to Ada's grounded fillers or a refusal.

For each (entity, role) we show:
  UNCONSTRAINED  — stock Qwen3 completes freely (it has never heard of 'erin' ->
                   it confabulates or hedges).
  GROUNDED       — same model + GroundedConstraint(memory.probe(entity, role)) ->
                   Ada's stored value, or a refusal when the fact is absent.

Small model on purpose (Qwen3-1.7B-4bit, cached) — the guarantee is architectural,
so the model size only affects fluency, not correctness. 7B/14B = change MODEL.

Run:  /opt/homebrew/anaconda3/bin/python run_qwen.py
"""
from __future__ import annotations

import copy

from mlx_lm import generate, load

from corpus import build_corpus
from memory import FactMemory
from hdc import bind, bundle
from lattice_adapter import GroundedConstraint, build_allowed

MODEL = "mlx-community/Qwen3-1.7B-4bit"
REFUSAL = "unknown"
MARGIN = 0.05
ADMIT = 0.10
MAXTOK = 8

# (subject entity, Ada role, natural completion prompt)
QUERIES = [
    ("erin", "spatial.location", "Erin lives in the city of"),
    ("erin", "spatial.origin", "Erin is originally from the state of"),
    ("helium", "entity.kind", "Helium is a kind of"),
    ("gold", "perceptual.color", "In this dataset, gold looks the color"),
    ("jack", "relational.object", "Jack works as a"),
]


def grounded_fillers(mem, entity, role):
    ranked = mem.probe(entity, role, topn=4)
    return [f for f, s in ranked if s >= ADMIT]


def complete(model, tok, prompt, constraint=None, max_tokens=MAXTOK):
    kw = dict(max_tokens=max_tokens, verbose=False)
    if constraint is not None:
        kw["logits_processors"] = [constraint]
    out = generate(model, tok, prompt=prompt, **kw)
    return out.strip().replace("\n", " ")


def main():
    print("distilling Ada facts into the HDC lattice...")
    c = build_corpus(repeat=1)
    mem = FactMemory(c["enc"]).distill(c["fillers"])
    print(f"  {len(mem.records)} entity records\n")

    print(f"loading stock model: {MODEL} ...")
    model, tok = load(MODEL)
    eos = tok.eos_token_id
    print("  loaded.\n")
    print("=" * 78)
    print("STOCK QWEN3 vs QWEN3 + Ada in-loop adapter")
    print("=" * 78)

    for entity, role, prompt in QUERIES:
        fillers = grounded_fillers(mem, entity, role)
        seqs, _ = build_allowed(tok, prompt, fillers, REFUSAL)
        con = GroundedConstraint(seqs, eos, None)

        free = complete(model, tok, prompt)
        grounded = complete(model, tok, prompt, constraint=con)
        print(f"\nQ: {prompt} ___     [Ada: {entity}.{role} -> {fillers or '∅'}]")
        print(f"   unconstrained Qwen : {free}")
        print(f"   Ada-grounded Qwen  : {grounded}")

    # refusal: delete erin's location from Ada; the adapter must refuse
    print("\n" + "=" * 78)
    print("REFUSAL — delete erin.spatial.location from Ada, ask again")
    print("=" * 78)
    mem2 = copy.deepcopy(mem)
    mem2.entity_facts["erin"] = [(r, f) for r, f in mem2.entity_facts["erin"]
                                 if r != "spatial.location"]
    mem2.records["erin"] = bundle([bind(mem2.enc.role_vec(r), mem2.enc.filler_vec(f))
                                   for r, f in mem2.entity_facts["erin"]])
    prompt = "Erin lives in the city of"
    fillers = grounded_fillers(mem2, "erin", "spatial.location")
    seqs, _ = build_allowed(tok, prompt, fillers, REFUSAL)
    con = GroundedConstraint(seqs, eos, None)
    print(f"\nQ: {prompt} ___     [Ada: erin.spatial.location -> {fillers or '∅ (deleted)'}]")
    print(f"   unconstrained Qwen : {complete(model, tok, prompt)}")
    print(f"   Ada-grounded Qwen  : {complete(model, tok, prompt, constraint=con)}  <- refuses, cannot invent")


if __name__ == "__main__":
    main()
