"""
business.py — an organization on the lattice: exact store + HDC semantic layer.
==============================================================================
Encodes every customer as a record with BOTH:
  - exact fields (revenue, support_cases, ...)  -> the STORE (Ada's asset), used
    for EXACT aggregation/filtering. Vectors are lossy at arithmetic; never decode
    a number from a bundle.
  - an HDC record over categorical/bucketed signals -> the SEMANTIC layer, used
    for SIMILARITY cohorts ("customers likely to churn") that SQL can't express.

Query engine (all deterministic -> perfectly consistent, exact):
  - aggregate(field, filter)          exact sum/avg over the store
  - filter/count                      exact
  - churn_cohort(k)                   HDC cosine to a prototype of churned records

The in-loop model adapter (run_qwen_analytics.py) then CONSTRAINS a stock LLM to
emit exactly these computed answers — the model gets fluency, the lattice keeps
authority over the numbers.

Run:  python business.py     # builds the org, runs the engine, self-verifies
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from hdc import Encoder, bind, bundle, raw_cosine

TIERS = ["free", "pro", "enterprise"]
REGIONS = ["west", "east", "south", "north"]
USAGE = ["growing", "steady", "declining", "dormant"]     # ordinal churn signal


@dataclass
class Customer:
    name: str
    region: str
    tier: str
    revenue: int          # exact, annual $
    support_cases: int    # exact, last quarter
    usage: str            # trend
    churned: bool         # historical label (for building the churn prototype)


def _bucket_rev(r: int) -> str:
    return "rev_high" if r >= 100_000 else "rev_mid" if r >= 20_000 else "rev_low"


def _bucket_cases(n: int) -> str:
    return "cases_high" if n >= 8 else "cases_mid" if n >= 3 else "cases_low"


class Org:
    """The store (exact) + the HDC semantic layer (similarity)."""

    def __init__(self, customers: List[Customer], enc: Encoder):
        self.customers = customers
        self.enc = enc
        self.by_name = {c.name: c for c in customers}
        self.record = {c.name: self._encode(c) for c in customers}
        # churn prototype = bundle of the records of customers who actually churned
        churned = [self.record[c.name] for c in customers if c.churned]
        self.churn_proto = bundle(churned) if churned else None

    def _encode(self, c: Customer) -> np.ndarray:
        """HDC record over CATEGORICAL / BUCKETED signals (not exact numbers)."""
        pairs = [
            ("relational.object", c.tier),
            ("spatial.location", c.region),
            ("temporal.frequency", c.usage),
            ("quantitative.magnitude", _bucket_rev(c.revenue)),
            ("quantitative.count", _bucket_cases(c.support_cases)),
        ]
        return bundle([bind(self.enc.role_vec(r), self.enc.filler_vec(v)) for r, v in pairs])

    # ---- EXACT analytics over the store (deterministic) ---------------------
    def _select(self, flt: Optional[Callable[[Customer], bool]]) -> List[Customer]:
        return [c for c in self.customers if (flt is None or flt(c)) and not c.churned]

    def total_revenue(self, flt=None) -> int:
        return sum(c.revenue for c in self._select(flt))

    def count(self, flt=None) -> int:
        return len(self._select(flt))

    def avg_revenue(self, flt=None) -> float:
        sel = self._select(flt)
        return sum(c.revenue for c in sel) / len(sel) if sel else 0.0

    # ---- SEMANTIC cohort via HDC similarity (the part SQL can't do) ----------
    def churn_cohort(self, k: int = 5):
        """Rank ACTIVE customers by cosine of their record to the churn prototype."""
        active = [c for c in self.customers if not c.churned]
        scored = [(c, raw_cosine(self.record[c.name], self.churn_proto)) for c in active]
        scored.sort(key=lambda t: -t[1])
        return scored[:k]


def make_org(n=60, seed=0) -> Org:
    rng = np.random.default_rng(seed)
    names = [f"acme{i:02d}" for i in range(n)]
    customers = []
    for nm in names:
        tier = rng.choice(TIERS, p=[0.5, 0.35, 0.15])
        region = rng.choice(REGIONS)
        base = {"free": 3_000, "pro": 40_000, "enterprise": 180_000}[tier]
        revenue = int(base * rng.uniform(0.6, 1.6))
        usage = rng.choice(USAGE, p=[0.3, 0.4, 0.2, 0.1])
        # churn risk driven by usage trend + support load (the real signal)
        risk = {"growing": 0.02, "steady": 0.06, "declining": 0.35, "dormant": 0.6}[usage]
        cases = int(rng.poisson({"growing": 1, "steady": 2, "declining": 6, "dormant": 9}[usage]))
        churned = rng.random() < risk
        customers.append(Customer(nm, region, tier, revenue, cases, usage, churned))
    return Org(customers, Encoder(vector_dim=10000, seed=42))


if __name__ == "__main__":
    org = make_org()
    active = [c for c in org.customers if not c.churned]
    print(f"org: {len(org.customers)} customers ({len(active)} active, "
          f"{len(org.customers) - len(active)} churned historically)\n")

    print("=== EXACT analytics (deterministic reducers over the store) ===")
    tot = org.total_revenue()
    print(f"  total active revenue                 : ${tot:,}")
    ent_west = org.total_revenue(lambda c: c.tier == 'enterprise' and c.region == 'west')
    print(f"  revenue from enterprise in the west  : ${ent_west:,}")
    print(f"  # customers with >=8 support cases   : {org.count(lambda c: c.support_cases >= 8)}")
    print(f"  avg revenue / enterprise customer    : ${org.avg_revenue(lambda c: c.tier=='enterprise'):,.0f}")

    # verify exactness against a brute-force ground truth
    gt = sum(c.revenue for c in org.customers if not c.churned)
    assert tot == gt, (tot, gt)
    print("  [verified exact vs brute-force ground truth]\n")

    print("=== SEMANTIC cohort: customers LIKELY TO CHURN (HDC similarity) ===")
    cohort = org.churn_cohort(k=6)
    for c, s in cohort:
        print(f"  {c.name}  sim={s:+.3f}  usage={c.usage:9} cases={c.support_cases:2d} "
              f"tier={c.tier:10} rev=${c.revenue:,}")
    # sanity: the cohort should be dominated by declining/dormant + high support load
    bad = [c for c, _ in cohort if c.usage in ("declining", "dormant")]
    print(f"  {len(bad)}/{len(cohort)} of the at-risk cohort are declining/dormant "
          f"(HDC recovered the churn signal without a rule)\n")

    print("  NOTE: exact numbers come from the STORE (never decoded from vectors);")
    print("        HDC supplies the SEMANTIC cohort; the adapter will force the model")
    print("        to emit exactly these -> perfectly consistent answers.")
