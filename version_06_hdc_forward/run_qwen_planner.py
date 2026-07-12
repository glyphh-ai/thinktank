"""
run_qwen_planner.py — NL question -> constrained typed plan -> exact answer (Qwen3).
==================================================================================
Closes the seam: the model turns a natural-language question into a query plan,
but EACH plan slot is constrained to its schema enum (GroundedConstraint), so the
model does the language mapping yet cannot emit an invalid/ungrounded plan. The
plan executes deterministically over the lattice, and the final answer is itself
constrained to the computed value -> consistent at BOTH layers (plan + answer).

Run:  /opt/homebrew/anaconda3/bin/python run_qwen_planner.py
"""
from __future__ import annotations

from mlx_lm import generate, load

from business import make_org
from plan import (Plan, execute, OP_OPTIONS, FIELD_OPTIONS, TIERS, REGIONS)
from lattice_adapter import GroundedConstraint, build_allowed

MODEL = "mlx-community/Qwen3-1.7B-4bit"
K = 3   # runs per question, to prove consistency


# few-shot exemplars per slot — teach the mapping AND the "any"/default behavior
SHOTS = {
    "op": ('operation', OP_OPTIONS, [
        ("What is the total revenue we bring in?", "total"),
        ("How much revenue from west-coast accounts?", "total"),
        ("How many customers are on the pro plan?", "count"),
        ("What's the average deal size?", "average"),
        ("Which account is most likely to stop being a customer?", "churn"),
        ("Who is at the highest risk of leaving?", "churn"),
        ("What is our monthly office rent?", "unsupported"),
    ]),
    "field": ('field being measured', FIELD_OPTIONS, [
        ("total annual revenue", "revenue"),
        ("how many support tickets were opened", "cases"),
    ]),
    "tier": ('customer tier it restricts to (use "any" if no tier is mentioned)', TIERS, [
        ("revenue from enterprise accounts", "enterprise"),
        ("how many customers do we have", "any"),
        ("total revenue across all customers", "any"),
    ]),
    "region": ('region it restricts to (use "any" if no region is mentioned)', REGIONS, [
        ("accounts on the west coast", "west"),
        ("total revenue overall", "any"),
        ("how many pro customers", "any"),
    ]),
}


def constrained_pick(model, tok, eos, question, slot):
    """Decode ONE plan slot, constrained to its enum -> a valid, grounded choice."""
    cue, options, shots = SHOTS[slot]
    ex = "\n".join(f'Q: "{q}"\n{cue} -> {a}' for q, a in shots)
    prompt = (f"Extract the {cue} for a customer-analytics query. "
              f"Only pick a specific value if the question explicitly mentions it.\n\n"
              f"{ex}\nQ: \"{question}\"\n{cue} ->")
    seqs, _ = build_allowed(tok, prompt, options, refusal=options[-1])
    con = GroundedConstraint(seqs, eos)
    out = generate(model, tok, prompt=prompt, max_tokens=6, verbose=False,
                   logits_processors=[con]).lower()
    for o in options:
        if o in out:
            return o
    return options[-1]


def plan_from_nl(model, tok, eos, question) -> Plan:
    op = constrained_pick(model, tok, eos, question, "op")
    if op in ("churn", "unsupported"):
        return Plan(op=op)
    field = constrained_pick(model, tok, eos, question, "field")
    tier = constrained_pick(model, tok, eos, question, "tier")
    region = constrained_pick(model, tok, eos, question, "region")
    return Plan(op=op, field=field, tier=tier, region=region)


def constrained_answer(model, tok, eos, question, result):
    prompt = f'Question: "{question}"\nThe answer is:'
    seqs, _ = build_allowed(tok, prompt, [result.answer], refusal="unsupported")
    con = GroundedConstraint(seqs, eos)
    return generate(model, tok, prompt=prompt, max_tokens=10, verbose=False,
                    logits_processors=[con]).strip().replace("\n", " ")


def main():
    org = make_org()
    model, tok = load(MODEL)
    eos = tok.eos_token_id

    questions = [
        "What is the total annual revenue across all active customers?",
        "How much revenue comes from enterprise customers on the west coast?",
        "How many pro-tier customers do we have?",
        "Which single customer is most likely to churn?",
        "What is the average revenue per enterprise customer?",
        "What is our total carbon footprint this year?",   # out of schema -> refuse
    ]

    print(f"model: {MODEL}   (plan slots + answer both constrained)\n")
    for q in questions:
        runs = []
        for _ in range(K):
            plan = plan_from_nl(model, tok, eos, q)
            res = execute(org, plan)
            ans = constrained_answer(model, tok, eos, q, res)
            runs.append((plan.describe(), res.display, ans))
        plans = {r[0] for r in runs}
        answers = {r[2] for r in runs}
        p0, disp0, a0 = runs[0]
        print("=" * 82)
        print(f"Q: {q}")
        print(f"   plan (model, constrained) : {p0}")
        print(f"   executed on lattice       : {disp0}")
        print(f"   final answer ({K}x)         : {a0}")
        print(f"   consistency               : {len(plans)} distinct plan, {len(answers)} distinct answer "
              f"across {K} runs {'✓' if len(plans) == 1 and len(answers) == 1 else '(varies)'}")


if __name__ == "__main__":
    main()
