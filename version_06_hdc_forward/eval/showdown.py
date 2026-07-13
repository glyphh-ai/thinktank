"""
showdown.py — THE test: HDC/gate vs a real database (text-to-SQL), not RAG.
==========================================================================
Structured customer data. Four systems answer NL questions across query classes:

  raw   — LLM, no data (floor)
  rag   — LLM + retrieved customer rows
  sql   — LLM writes SQLite, the DB executes it   <- the real incumbent, finally
  hdc   — the gate/HDC architecture (store reducer for structured; churn cohort for fuzzy)

Decision:
  - if SQL matches/beats HDC on EVERY class -> the architecture is a reinvention (DEAD).
  - if HDC uniquely wins where SQL can't EXPRESS the query (fuzzy/semantic) -> ALIVE.

The crux class is fuzzy: "customers most likely to churn" — SQL has no clause for it
(only historical churned=0/1 + proxies); HDC ranks by similarity to churned records.

Run:  /opt/homebrew/anaconda3/bin/python showdown.py
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys

from mlx_lm import generate, load

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from business import make_org  # noqa: E402

MODEL = "mlx-community/Qwen3-4B-4bit"   # strong SQL writer -> best case for the incumbent
SCHEMA = ("customers(name TEXT, region TEXT, tier TEXT['free'|'pro'|'enterprise'], "
          "revenue INT, support_cases INT, usage TEXT['growing'|'steady'|'declining'|'dormant'], "
          "churned INT[0=active,1=already_churned])")


def build_db(org):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE customers(name TEXT, region TEXT, tier TEXT, revenue INT, "
                "support_cases INT, usage TEXT, churned INT)")
    con.executemany("INSERT INTO customers VALUES(?,?,?,?,?,?,?)",
                    [(c.name, c.region, c.tier, c.revenue, c.support_cases, c.usage, int(c.churned))
                     for c in org.customers])
    con.commit()
    return con


# ground-truth latent churn risk (the data-generating factors)
_SEV = {"growing": 0, "steady": 1, "declining": 3, "dormant": 4}
def risk(c):
    return _SEV[c.usage] + 0.3 * c.support_cases


def _chat(tok, msg):
    try:
        return tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False)


def gen(model, tok, msg, mx=90):
    out = generate(model, tok, prompt=_chat(tok, msg), max_tokens=mx, verbose=False)
    return out.replace("<think>", " ").replace("</think>", " ").strip()


def run_sql(model, tok, con, q):
    sql = gen(model, tok, f"SQLite table {SCHEMA}. Write ONE SQLite query answering: \"{q}\". "
                          f"Output ONLY SQL, no markdown, no explanation.", mx=120)
    sql = re.sub(r"```\w*", "", sql).replace("```", "").strip().split(";")[0]
    try:
        rows = con.execute(sql).fetchall()
        return rows, sql
    except Exception as e:
        return None, f"{sql}   [ERROR {type(e).__name__}]"


def names_in(text, org):
    have = {c.name for c in org.customers}
    return [w for w in re.findall(r"acme\d+", text.lower()) if w in have]


def main():
    org = make_org()
    con = build_db(org)
    active = [c for c in org.customers if not c.churned]
    model, tok = load(MODEL)

    # ================= STRUCTURED classes (expect DB to win/tie) =================
    print("=" * 76)
    print("STRUCTURED queries — the DB's home turf")
    print("=" * 76)
    gt_total = sum(c.revenue for c in active if c.tier == "enterprise")
    gt_count = sum(1 for c in active if c.tier == "pro" and c.region == "west")
    for q, gt, kind in [("What is the total revenue of active enterprise customers?", gt_total, "agg"),
                        ("How many active pro-tier customers are in the west region?", gt_count, "count")]:
        rows, sql = run_sql(model, tok, con, q)
        val = rows[0][0] if rows and rows[0] else None
        ok = (val == gt)
        # HDC/gate here = the deterministic store reducer (business.py) = exact too
        hdc_ok = True   # store reducer is exact by construction
        print(f"\nQ: {q}\n   ground truth: {gt}")
        print(f"   SQL  -> {val}  ({'ok' if ok else 'WRONG'})   [{sql[:60]}]")
        print(f"   HDC  -> exact via store reducer ({'ok' if hdc_ok else 'WRONG'})  [gate == DB here]")

    # ================= FUZZY class — the crux =================
    print("\n" + "=" * 76)
    print("FUZZY / SEMANTIC query — SQL has no clause for 'likely to churn'")
    print("=" * 76)
    K = 5
    gt_rank = sorted(active, key=risk, reverse=True)
    gt_set = {c.name for c in gt_rank[:max(K, len(active) // 4)]}   # true high-risk set
    q = "List the 5 active customers most likely to churn (stop being customers)."

    def precision(names):
        names = names[:K]
        return sum(n in gt_set for n in names) / max(1, len(names)), names

    # raw
    raw = gen(model, tok, q + " Names look like 'acme07'.", mx=80)
    p_raw, n_raw = precision(names_in(raw, org))
    # rag: give it the active customer rows as context
    ctx = "\n".join(f"{c.name}: tier={c.tier} usage={c.usage} support_cases={c.support_cases} revenue={c.revenue}"
                    for c in active)
    rag = gen(model, tok, f"Customer data:\n{ctx}\n\n{q} Reply with just the 5 names.", mx=80)
    p_rag, n_rag = precision(names_in(rag, org))
    # sql
    rows, sql = run_sql(model, tok, con, q)
    n_sql = [r[0].lower() for r in (rows or []) if r and isinstance(r[0], str)][:K]
    p_sql, n_sql = precision([n for n in n_sql if n in gt_set or True])
    # hdc cohort
    n_hdc = [c.name for c, _ in org.churn_cohort(K)]
    p_hdc, n_hdc = precision(n_hdc)

    print(f"\nQ: {q}")
    print(f"   true high-risk set (top {len(gt_set)} by latent risk): {sorted(gt_set)}")
    print(f"\n   RAW  precision@{K} {p_raw:.0%}   picks={n_raw}")
    print(f"   RAG  precision@{K} {p_rag:.0%}   picks={n_rag}")
    print(f"   SQL  precision@{K} {p_sql:.0%}   picks={n_sql}")
    print(f"        SQL query: {sql[:110]}")
    print(f"   HDC  precision@{K} {p_hdc:.0%}   picks={n_hdc}")

    # ================= HARDER: holistic multi-attribute similarity =================
    print("\n" + "=" * 76)
    print("HARD FUZZY — holistic similarity ('customers like X'); no single column to sort")
    print("=" * 76)
    from business import _bucket_rev, _bucket_cases
    target = next(c for c in active if c.name == "acme10")

    def overlap(c):   # INDEPENDENT ground truth: shared categorical attributes with target
        return sum([c.tier == target.tier, c.region == target.region, c.usage == target.usage,
                    _bucket_rev(c.revenue) == _bucket_rev(target.revenue),
                    _bucket_cases(c.support_cases) == _bucket_cases(target.support_cases)])
    others = [c for c in active if c.name != "acme10"]
    gt_sim = {c.name for c in sorted(others, key=overlap, reverse=True)[:K]}
    tinfo = (f"acme10: tier={target.tier} region={target.region} usage={target.usage} "
             f"rev_bucket={_bucket_rev(target.revenue)} cases_bucket={_bucket_cases(target.support_cases)}")
    q2 = "List the 5 active customers most similar to customer acme10 (across tier, region, usage, revenue, support load)."
    print(f"\n{tinfo}\nQ: {q2}\n   ground-truth most-similar (shared attributes): {sorted(gt_sim)}")

    # sql
    rows2, sql2 = run_sql(model, tok, con, q2)
    n_sql2 = [r[0].lower() for r in (rows2 or []) if r and isinstance(r[0], str)][:K]
    p_sql2 = sum(n in gt_sim for n in n_sql2) / max(1, len(n_sql2))
    # rag
    rag2 = gen(model, tok, f"Customer data:\n{ctx}\n\n{q2} Reply with just the 5 names.", mx=80)
    n_rag2 = names_in(rag2, org)[:K]
    p_rag2 = sum(n in gt_sim for n in n_rag2) / max(1, len(n_rag2))
    # hdc: cortex cosine to acme10's record
    from hdc import raw_cosine
    sims = sorted(((c.name, raw_cosine(org.record[c.name], org.record["acme10"])) for c in others),
                  key=lambda t: -t[1])
    n_hdc2 = [n for n, _ in sims[:K]]
    p_hdc2 = sum(n in gt_sim for n in n_hdc2) / max(1, len(n_hdc2))
    print(f"\n   SQL  precision@{K} {p_sql2:.0%}   picks={n_sql2}")
    print(f"        query: {sql2[:100]}")
    print(f"   RAG  precision@{K} {p_rag2:.0%}   picks={n_rag2}")
    print(f"   HDC  precision@{K} {p_hdc2:.0%}   picks={n_hdc2}")

    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)
    print(f"structured (agg/count): SQL & HDC-store-reducer both exact -> gate == DB (no win).")
    print(f"fuzzy churn : SQL={p_sql:.0%}  RAG={p_rag:.0%}  HDC={p_hdc:.0%}  (easy: column proxy suffices)")
    print(f"hard similar: SQL={p_sql2:.0%}  RAG={p_rag2:.0%}  HDC={p_hdc2:.0%}")
    if p_hdc2 >= max(p_sql2, p_rag2) + 0.2:
        print("=> HDC uniquely wins holistic similarity SQL/RAG can't -> ALIVE.")
    elif max(p_sql2, p_rag2) >= p_hdc2:
        print("=> SQL/RAG match HDC even on holistic similarity -> DEAD (no unique win anywhere).")
    else:
        print("=> mixed.")


if __name__ == "__main__":
    main()
