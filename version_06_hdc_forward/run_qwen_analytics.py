"""
run_qwen_analytics.py — consistency test: analytical queries, in the loop.
==========================================================================
Answers the question directly: does wiring a stock LLM to the lattice make it
answer complex analytical questions MORE CONSISTENTLY?

For each query we compute the exact/cohort answer deterministically (business.py),
then run stock Qwen3 K times two ways:
  UNCONSTRAINED (sampled)  -> the model guesses; different (often wrong) each run.
  ADAPTER-CONSTRAINED      -> logits masked to the computed answer; identical every
                              run, and correct.

Consistency isn't the model reasoning better — it's the lattice keeping authority
over the answer. Exact numbers come from the STORE; the churn cohort from HDC.

Run:  /opt/homebrew/anaconda3/bin/python run_qwen_analytics.py
"""
from __future__ import annotations

import re
from collections import Counter

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from business import make_org
from lattice_adapter import GroundedConstraint, build_allowed

MODEL = "mlx-community/Qwen3-1.7B-4bit"
K = 5
TEMP = 0.8


def gen(model, tok, prompt, sampler=None, constraint=None, max_tokens=12):
    kw = dict(max_tokens=max_tokens, verbose=False)
    if sampler is not None:
        kw["sampler"] = sampler
    if constraint is not None:
        kw["logits_processors"] = [constraint]
    return generate(model, tok, prompt=prompt, **kw).strip().replace("\n", " ")


def first_int(s):
    m = re.search(r"-?\d[\d,]*", s)
    return int(m.group().replace(",", "")) if m else None


def main():
    org = make_org()
    total = org.total_revenue()
    ent_west = org.total_revenue(lambda c: c.tier == "enterprise" and c.region == "west")
    top_churn = org.churn_cohort(k=1)[0][0].name

    model, tok = load(MODEL)
    eos = tok.eos_token_id
    sampler = make_sampler(temp=TEMP)

    queries = [
        ("Across all active customers, the total annual revenue in dollars is",
         str(total), "int"),
        ("The total annual revenue from enterprise-tier customers in the west region, in dollars, is",
         str(ent_west), "int"),
        ("Based on usage and support load, the single customer most likely to churn is",
         top_churn, "name"),
    ]

    print(f"model: {MODEL}   K={K} samples/query   temp={TEMP}\n")
    print(f"[ground truth from lattice]  total=${total:,}  ent_west=${ent_west:,}  top_churn={top_churn}\n")

    for prompt, answer, kind in queries:
        # unconstrained: sample K times
        free = [gen(model, tok, prompt, sampler=sampler) for _ in range(K)]
        if kind == "int":
            free_vals = [first_int(x) for x in free]
            free_keys = [str(v) for v in free_vals]
            correct_free = sum(v == int(answer) for v in free_vals)
        else:
            free_keys = [x.lower() for x in free]
            correct_free = sum(answer.lower() in x.lower() for x in free)
        uniq_free = len(set(free_keys))

        # constrained: mask to the computed answer, K times
        con_out = []
        for _ in range(K):
            seqs, _ = build_allowed(tok, prompt, [answer], refusal="unknown")
            con = GroundedConstraint(seqs, eos)
            con_out.append(gen(model, tok, prompt, constraint=con))
        uniq_con = len(set(con_out))
        correct_con = sum(answer.lower() in o.lower() for o in con_out)

        print("=" * 80)
        print(f"Q: {prompt} ___")
        print(f"   lattice answer: {answer}")
        print(f"   UNCONSTRAINED  ({K}x): {free}")
        print(f"      -> {uniq_free} distinct answers, {correct_free}/{K} correct")
        print(f"   ADAPTER        ({K}x): {con_out}")
        print(f"      -> {uniq_con} distinct answer, {correct_con}/{K} correct")

    print("\n" + "=" * 80)
    print("Consistency comes from the lattice holding authority over the answer,")
    print("not from the model reasoning better. Exact = store; cohort = HDC; adapter = guarantee.")


if __name__ == "__main__":
    main()
