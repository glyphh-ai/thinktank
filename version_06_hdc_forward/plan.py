"""
plan.py — NL question -> TYPED, GROUNDED query plan -> deterministic execution.
==============================================================================
The model's job is language: turn a question into a plan. But the plan is
confined to a schema of real ops/fields/values (run_qwen_planner.py constrains
each slot to its enum), so the model CANNOT reference a field Ada doesn't have —
an out-of-schema request routes to refusal. Execution is deterministic over the
lattice (exact store + HDC cohort), so the answer is exact and consistent.

This module is the schema + executor + a CPU self-test with an oracle planner
(no model needed). The constrained slot decoding lives in run_qwen_planner.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from business import make_org, Org

# ── the plan schema (small typed DSL) ────────────────────────────────────────
OPS = {           # op token -> (kind, aggregation)
    "total":   ("aggregate", "sum"),
    "average": ("aggregate", "avg"),
    "count":   ("count", None),
    "churn":   ("cohort", None),
    "unsupported": ("refuse", None),      # escape hatch for out-of-schema questions
}
FIELDS = {"revenue": "revenue", "cases": "support_cases"}
TIERS = ["enterprise", "pro", "free", "any"]
REGIONS = ["west", "east", "south", "north", "any"]

OP_OPTIONS = list(OPS.keys())
FIELD_OPTIONS = list(FIELDS.keys())


@dataclass
class Plan:
    op: str = "unsupported"
    field: str = "revenue"
    tier: str = "any"
    region: str = "any"
    k: int = 1

    def filter(self):
        conds = []
        if self.tier != "any":
            conds.append(lambda c, t=self.tier: c.tier == t)
        if self.region != "any":
            conds.append(lambda c, r=self.region: c.region == r)
        if not conds:
            return None
        return lambda c: all(f(c) for f in conds)

    def describe(self):
        where = " ".join(x for x in [
            "" if self.tier == "any" else f"tier={self.tier}",
            "" if self.region == "any" else f"region={self.region}"] if x)
        return f"{self.op}({self.field}{' where ' + where if where else ''})"


@dataclass
class Result:
    kind: str            # "number" | "name" | "refuse"
    answer: str          # the canonical answer string the adapter will constrain to
    display: str         # human-readable


def execute(org: Org, plan: Plan) -> Result:
    kind, agg = OPS.get(plan.op, ("refuse", None))
    if kind == "refuse":
        return Result("refuse", "unsupported", "unsupported (out of schema)")
    field = FIELDS[plan.field]
    flt = plan.filter()

    if kind == "cohort":
        top = org.churn_cohort(k=max(1, plan.k))
        name = top[0][0].name
        return Result("name", name, f"most likely to churn: {name}")

    sel = [c for c in org.customers if (flt is None or flt(c)) and not c.churned]
    vals = [getattr(c, field) for c in sel]
    if kind == "count":
        return Result("number", str(len(sel)), f"count = {len(sel)}")
    if not vals:
        return Result("number", "0", "0 (no matching customers)")
    if agg == "sum":
        v = int(sum(vals))
    else:  # avg
        v = int(round(sum(vals) / len(vals)))
    return Result("number", str(v), f"{plan.op} {plan.field}{' (filtered)' if flt else ''} = {v}")


if __name__ == "__main__":
    org = make_org()

    # oracle plans (what a correct planner should produce) -> exact answers,
    # cross-checked against brute-force ground truth.
    cases = [
        ("total revenue, all customers",
         Plan("total", "revenue"),
         sum(c.revenue for c in org.customers if not c.churned)),
        ("total revenue, enterprise in the west",
         Plan("total", "revenue", "enterprise", "west"),
         sum(c.revenue for c in org.customers if not c.churned and c.tier == "enterprise" and c.region == "west")),
        ("count of pro customers",
         Plan("count", "revenue", "pro"),
         len([c for c in org.customers if not c.churned and c.tier == "pro"])),
        ("average revenue of enterprise customers",
         Plan("average", "revenue", "enterprise"),
         int(round(sum(c.revenue for c in org.customers if not c.churned and c.tier == "enterprise")
                   / max(1, len([c for c in org.customers if not c.churned and c.tier == "enterprise"]))))),
    ]
    print("=== plan execution vs brute-force ground truth ===")
    for desc, plan, gt in cases:
        r = execute(org, plan)
        ok = r.answer == str(gt)
        print(f"  {desc:42} {plan.describe():42} -> {r.answer:>10}  {'ok' if ok else f'WRONG (gt {gt})'}")
        assert ok, (desc, r.answer, gt)

    # cohort + refusal
    print(f"  {'most likely to churn':42} {'churn(k=1)':42} -> {execute(org, Plan('churn')).answer}")
    print(f"  {'out-of-schema (e.g. profit margin)':42} {'unsupported':42} -> {execute(org, Plan('unsupported')).answer}")
    print("\nplan.py self-test PASSED — typed plans execute exactly; out-of-schema refuses")
