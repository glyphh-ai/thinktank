"""
schema_guard.py — semantic schema-membership: refuse out-of-schema questions.
=============================================================================
The tiny LLM's `op=unsupported` pick was unstable (whack-a-mole). Our RANDOM-atom
HDC can't help here — it has no semantic content ("revenue" ⟂ "earnings" ⟂ "carbon").
Semantic membership needs SEMANTIC embeddings, so we use the cached
Qwen3-Embedding-0.6B via sentence-transformers.

Design: a nearest-prototype router with an explicit OUT_OF_SCHEMA negative class.
Absolute cosine doesn't separate (everything business-y scores ~0.6 to 'revenue');
the robust signal is the MARGIN between the best in-schema field and the
OUT_OF_SCHEMA prototype. Real queries keep a clear gap; out-of-schema questions
collapse to ~0 -> refuse.

Bonus: the router also nails the metric/intent (revenue / support_cases / churn),
which is exactly what the 1.7B kept misclassifying — so embeddings do the SEMANTIC
routing and the LLM is left with only structured slot extraction.

Run:  /opt/homebrew/anaconda3/bin/python schema_guard.py
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")

FIELD_PROTOS: Dict[str, List[str]] = {
    "revenue": ["revenue", "annual revenue", "total sales", "how much money customers pay",
                "average deal size", "spend per account", "revenue from enterprise customers",
                "revenue from customers in a region", "sales from a customer segment"],
    "support_cases": ["support cases", "support tickets opened", "customer complaints",
                      "number of support requests"],
    "churn": ["churn", "customers likely to leave", "accounts at risk of cancelling",
              "who will stop being a customer", "retention risk"],
}
OOS_PROTOS = ["carbon footprint", "profit margin", "number of employees", "headcount",
              "office rent", "marketing spend", "website traffic", "stock price",
              "server costs", "the weather"]

THRESHOLD = 0.05     # min (best_field - out_of_schema) cosine margin to be in-schema


@dataclass
class Route:
    in_schema: bool
    field: str          # revenue | support_cases | churn (meaningful only if in_schema)
    margin: float
    field_sim: float
    oos_sim: float


class SchemaGuard:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model_name)
        # nearest-EXEMPLAR (max-pool) routing: keep every phrase's vector, not a
        # blurred mean — sharper on specific near-matches (e.g. 'headcount').
        self.field_names, self.field_exe = [], []
        for name, phrases in FIELD_PROTOS.items():
            v = self.m.encode(phrases, normalize_embeddings=True)
            self.field_exe.append(v); self.field_names.append(name)
        self.oos_exe = self.m.encode(OOS_PROTOS, normalize_embeddings=True)

    def classify(self, question: str) -> Route:
        e = self.m.encode([question], normalize_embeddings=True)[0]
        field_best = [float((exe @ e).max()) for exe in self.field_exe]
        i = int(np.argmax(field_best))
        field_sim = field_best[i]
        oos_sim = float((self.oos_exe @ e).max())
        margin = field_sim - oos_sim
        return Route(in_schema=margin >= THRESHOLD, field=self.field_names[i],
                     margin=margin, field_sim=field_sim, oos_sim=oos_sim)


if __name__ == "__main__":
    g = SchemaGuard()
    tests = [
        ("what is the total revenue across all customers", True, "revenue"),
        ("how much revenue from enterprise on the west coast", True, "revenue"),
        ("average revenue per enterprise customer", True, "revenue"),
        ("which customer is most likely to churn", True, "churn"),
        ("how many support tickets were opened", True, "support_cases"),
        ("what is our total carbon footprint this year", False, None),
        ("how many employees does each customer have", False, None),
        ("what is our office rent", False, None),
    ]
    print(f"{'verdict':8} {'field':14} {'margin':>7}  question")
    ok = 0
    for q, want_in, want_field in tests:
        r = g.classify(q)
        verdict = "IN" if r.in_schema else "REFUSE"
        hit = (r.in_schema == want_in) and (not want_in or r.field == want_field)
        ok += hit
        flag = "" if hit else "  <-- MISS"
        shown = r.field if r.in_schema else "-"
        print(f"{verdict:8} {shown:14} {r.margin:+7.3f}  {q}{flag}")
    print(f"\n{ok}/{len(tests)} correct  (in/out-of-schema routing + metric)")
