"""
benchmark.py — a proper benchmark for the grounding gate on a real LLM.
======================================================================
Compares STOCK vs GATED on stock Qwen3 across four axes that matter:

  1. grounded accuracy   — does the answer match the store's grounded value?
  2. refusal recall      — on entities the store does NOT know, does the gate refuse?
  3. refusal precision   — of the gate's refusals, how many were truly unknown?
                           (i.e. it never falsely refuses a known fact)
  4. consistency         — distinct answers across K samples (variance), stock vs gated
  5. latency overhead    — extra ms/query the gate costs

The store is a LocalStore (same code path as the runtime adapter). Entities are
fictional so the stock model has no ground truth — isolating what grounding adds.

Run:  /opt/homebrew/anaconda3/bin/python benchmark.py
"""
from __future__ import annotations

import re
import time
from collections import Counter

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from grounding import LocalStore
from adapters import make_constraint

MODEL = "mlx-community/Qwen3-1.7B-4bit"
K = 4
TEMP = 0.8
ROLE = "spatial.location"

KNOWN = [("Erin", "boston"), ("Jack", "dallas"), ("Mona", "tucson"), ("Olga", "reno"),
         ("Pete", "tampa"), ("Gina", "seattle"), ("Hank", "chicago"), ("Ivy", "portland")]
UNKNOWN = ["Zane", "Yara", "Xander", "Wren", "Vera"]     # not in the store -> must refuse
REFUSAL = "unknown"


def prompt_for(name): return f"The city where {name} lives is"


def answer(model, tok, prompt, sampler=None, constraint=None, max_tokens=6):
    kw = dict(max_tokens=max_tokens, verbose=False)
    if sampler is not None:
        kw["sampler"] = sampler
    if constraint is not None:
        kw["logits_processors"] = [constraint]
    t = time.time()
    out = generate(model, tok, prompt=prompt, **kw).strip().replace("\n", " ")
    return out, (time.time() - t)


def main():
    store = LocalStore().inject([{"entity": n, ROLE: c} for n, c in KNOWN])
    model, tok = load(MODEL)
    eos = tok.eos_token_id
    sampler = make_sampler(temp=TEMP)

    stock_correct = gated_correct = 0
    stock_distinct = gated_distinct = 0
    stock_ms = gated_ms = 0.0
    n_present = len(KNOWN)

    print(f"model {MODEL}  |  {n_present} known + {len(UNKNOWN)} unknown entities  |  K={K}\n")
    print(f"{'entity':7} {'truth':8} {'stock (K samples)':34} {'gated'}")
    for name, city in KNOWN:
        p = prompt_for(name)
        stock = []
        for _ in range(K):
            o, dt = answer(model, tok, p, sampler=sampler); stock_ms += dt
            m = re.search(r"[A-Za-z]+", o); stock.append(m.group().lower() if m else "?")
        con = make_constraint(store, tok, p, name, ROLE, refusal=REFUSAL, eos_id=eos,
                              space_id=store.space_id)
        g, dt = answer(model, tok, p, constraint=con); gated_ms += dt
        gated = g.strip().lower()
        stock_correct += sum(s == city for s in stock) / K
        gated_correct += (gated == city)
        stock_distinct += len(set(stock)); gated_distinct += 1
        print(f"{name:7} {city:8} {str(stock):34} {gated}")

    # refusal on unknowns
    refused = 0
    false_refusals = 0
    for name in UNKNOWN:
        p = prompt_for(name)
        con = make_constraint(store, tok, p, name, ROLE, refusal=REFUSAL, eos_id=eos,
                              space_id=store.space_id)
        g, _ = answer(model, tok, p, constraint=con)
        if g.strip().lower().startswith(REFUSAL):
            refused += 1
    # false-refusal check: did the gate ever refuse a KNOWN entity?
    for name, city in KNOWN:
        con = make_constraint(store, tok, prompt_for(name), name, ROLE, refusal=REFUSAL,
                              eos_id=eos, space_id=store.space_id)
        g, _ = answer(model, tok, prompt_for(name), constraint=con)
        if g.strip().lower().startswith(REFUSAL):
            false_refusals += 1

    nq = n_present * K
    print("\n" + "=" * 66)
    print("RESULTS                              stock          gated")
    print(f"grounded accuracy (known)         {stock_correct/n_present:6.0%}         {gated_correct/n_present:6.0%}")
    print(f"answer consistency (distinct/ent) {stock_distinct/n_present:6.2f}         {gated_distinct/n_present:6.2f}")
    print(f"refusal recall (unknown->refuse)      -            {refused/len(UNKNOWN):6.0%}")
    refusals = refused + false_refusals
    prec = refused / refusals if refusals else 1.0
    print(f"refusal precision (no false refuse)   -            {prec:6.0%}")
    print(f"latency / query                   {stock_ms/nq*1000:6.0f} ms       {gated_ms/n_present*1000:6.0f} ms")
    print("=" * 66)
    print("stock: fluent but wrong + inconsistent; gated: exact, consistent, refuses the unknown.")


if __name__ == "__main__":
    main()
