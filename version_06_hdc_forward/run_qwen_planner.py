"""
run_qwen_planner.py — NL question -> constrained typed plan -> exact answer.
==========================================================================
Division of labor along each component's strength:
  SEMANTIC routing (metric / churn / out-of-schema)  -> embeddings (schema_guard),
    size-independent + robust. Refuses out-of-schema deterministically.
  STRUCTURED extraction (aggregation + tier/region filters) -> the LLM, each slot
    constrained to its schema enum so it can't emit an invalid plan.
  EXECUTION + ANSWER -> deterministic over the lattice, answer constrained.

This fixes the earlier op-intent whack-a-mole: the fragile semantic decisions
(is this churn? is this out-of-schema?) move to the embedding router; the LLM is
left with the extraction it's reliable at.

Run:  /opt/homebrew/anaconda3/bin/python run_qwen_planner.py
"""
from __future__ import annotations

from mlx_lm import generate, load

from business import make_org
from plan import Plan, execute, TIERS, REGIONS
from schema_guard import SchemaGuard
from lattice_adapter import GroundedConstraint, build_allowed

MODEL = "mlx-community/Qwen3-1.7B-4bit"
K = 3
AGG_OPTIONS = ["total", "average", "count"]

# few-shot exemplars for the STRUCTURED slots the LLM still owns
SHOTS = {
    "agg": ('aggregation', AGG_OPTIONS, [
        ("What is the total revenue we bring in?", "total"),
        ("What's the average deal size per account?", "average"),
        ("How many customers are on the pro plan?", "count"),
    ]),
    "tier": ('customer tier it restricts to (use "any" if none is mentioned)', TIERS, [
        ("revenue from enterprise accounts", "enterprise"),
        ("how many pro-tier customers do we have", "pro"),
        ("free plan users", "free"),
        ("total revenue across all customers", "any"),
    ]),
    "region": ('region it restricts to (use "any" if none is mentioned)', REGIONS, [
        ("accounts on the west coast", "west"),
        ("customers in the northern region", "north"),
        ("total revenue overall", "any"),
    ]),
}


def constrained_pick(model, tok, eos, question, slot):
    cue, options, shots = SHOTS[slot]
    ex = "\n".join(f'Q: "{q}"\n{cue} -> {a}' for q, a in shots)
    prompt = (f"Extract the {cue} for a customer-analytics query. "
              f"Only pick a specific value if the question explicitly mentions it.\n\n"
              f"{ex}\nQ: \"{question}\"\n{cue} ->")
    seqs, _ = build_allowed(tok, prompt, options, refusal=options[-1])
    con = GroundedConstraint(seqs, eos)
    out = generate(model, tok, prompt=prompt, max_tokens=6, verbose=False,
                   logits_processors=[con]).lower()
    return next((o for o in options if o in out), options[-1])


def plan_from_nl(guard, model, tok, eos, question) -> Plan:
    route = guard.classify(question)
    if not route.in_schema:
        return Plan(op="unsupported")                 # embedding router refuses
    if route.field == "churn":
        return Plan(op="churn")
    field = "revenue" if route.field == "revenue" else "cases"
    agg = constrained_pick(model, tok, eos, question, "agg")
    tier = constrained_pick(model, tok, eos, question, "tier")
    region = constrained_pick(model, tok, eos, question, "region")
    return Plan(op=agg, field=field, tier=tier, region=region)


def constrained_answer(model, tok, eos, question, result):
    prompt = f'Question: "{question}"\nThe answer is:'
    seqs, _ = build_allowed(tok, prompt, [result.answer], refusal="unsupported")
    con = GroundedConstraint(seqs, eos)
    return generate(model, tok, prompt=prompt, max_tokens=10, verbose=False,
                    logits_processors=[con]).strip().replace("\n", " ")


def main():
    org = make_org()
    print("loading embedding router + LLM...")
    guard = SchemaGuard()
    model, tok = load(MODEL)
    eos = tok.eos_token_id

    questions = [
        "What is the total annual revenue across all active customers?",
        "How much revenue comes from enterprise customers on the west coast?",
        "How many pro-tier customers do we have?",
        "Which single customer is most likely to churn?",
        "What is the average revenue per enterprise customer?",
        "What is our total carbon footprint this year?",     # out of schema -> refuse
        "How many support tickets came from enterprise accounts?",
    ]
    print(f"model: {MODEL}  |  router: Qwen3-Embedding-0.6B  |  K={K}\n")
    for q in questions:
        runs = []
        for _ in range(K):
            plan = plan_from_nl(guard, model, tok, eos, q)
            res = execute(org, plan)
            ans = constrained_answer(model, tok, eos, q, res)
            runs.append((plan.describe(), res.display, ans))
        plans = {r[0] for r in runs}; answers = {r[2] for r in runs}
        p0, disp0, a0 = runs[0]
        print("=" * 84)
        print(f"Q: {q}")
        print(f"   plan (router+LLM, constrained): {p0}")
        print(f"   executed on lattice           : {disp0}")
        print(f"   final answer ({K}x)             : {a0}   "
              f"[{len(plans)} plan / {len(answers)} answer {'✓' if len(plans)==len(answers)==1 else 'varies'}]")


if __name__ == "__main__":
    main()
