"""
Strat 2 — solve rate
====================
Strat 1 showed every looped run is binary: solved (perfect to depth 32)
or stuck near the shortcut floor. So measure P(solve) directly.

B, C, C0 x n in {128, 256, 512, 1024} x 20 fresh seeds (100..119).
Solved := accuracy at D=16 >= 0.99. A is dropped: 0/12 solves in strat 1
at every size up to 8192.

Run:  python -u strat2.py      (~25 min CPU)
Out:  strat2_results.json + solve table + gates
"""

import json
from math import comb

import numpy as np

from models import build, random_bank, train_bank
from run import LR, STEPS, accuracy, eval_sets, fit, router_hits, train_set

MODELS = ["B", "C", "C0"]
SIZES = [128, 256, 512, 1024]
SEEDS = list(range(100, 120))
SOLVED = 0.99


def fisher_greater(a, n1, b, n2):
    """One-sided Fisher exact p: P(first group has >= a successes | margins)."""
    k = a + b
    total = comb(n1 + n2, k)
    return sum(comb(n1, i) * comb(n2, k - i)
               for i in range(a, min(n1, k) + 1)) / total


def report(rows):
    solved = {(m, n): sum(r["acc"]["16"] >= SOLVED for r in rows if r["model"] == m and r["n"] == n)
              for m in MODELS for n in SIZES}
    print("\nseeds solved (D=16 >= 0.99), of 20")
    print("n      " + "".join(f"{m:>6s}" for m in MODELS) + "   C>B one-sided p")
    for n in SIZES:
        p = fisher_greater(solved["C", n], 20, solved["B", n], 20)
        print(f"{n:<6d} " + "".join(f"{solved[m, n]:6d}" for m in MODELS) + f"   {p:.3f}")
    tot = {m: sum(solved[m, n] for n in SIZES) for m in MODELS}
    N = 20 * len(SIZES)
    print("pooled " + "".join(f"{tot[m]:6d}" for m in MODELS) + f"   (of {N})")

    p_cb = fisher_greater(tot["C"], N, tot["B"], N)
    p_cc0 = fisher_greater(tot["C"], N, tot["C0"], N)
    d_cb = (tot["C"] - tot["B"]) / N
    d_cc0 = (tot["C"] - tot["C0"]) / N
    hits = [r["router_hits"] for r in rows if r["model"] == "C" and r["acc"]["16"] >= SOLVED]
    gates = {
        "S1 CORE C solves more often than B (pooled): rate diff >= 0.15 and p < 0.05":
            (d_cb >= 0.15 and p_cb < 0.05, f"diff {d_cb:+.3f}, p {p_cb:.4f}"),
        "S2 content not shape: C - C0 pooled rate >= 0.15 and p < 0.05":
            (d_cc0 >= 0.15 and p_cc0 < 0.05, f"diff {d_cc0:+.3f}, p {p_cc0:.2e}"),
        "S3 every solved C run routes >= 9/10 ops to its own tiny model":
            (bool(hits) and min(hits) >= 9, f"{len(hits)} solved, min hits {min(hits) if hits else '-'}"),
    }
    print("\nGATES")
    for name, (ok, detail) in gates.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}   [{detail}]")
    verdict = ("ALIVE" if gates[next(iter(gates))][0]
               else "DEAD" if tot["B"] >= tot["C"] else "INCONCLUSIVE")
    print(f"\nVERDICT (core claim, rung 1): {verdict}")


if __name__ == "__main__":
    bank = train_bank()
    evals = eval_sets()
    rows = []
    for n in SIZES:
        for seed in SEEDS:
            data = train_set(n, seed)
            for name in MODELS:
                model = build(name, seed, bank if name == "C" else random_bank(seed))
                fit(model, data, LR[name], seed, STEPS)
                acc = accuracy(model, evals)
                row = dict(model=name, n=n, seed=seed, acc={str(d): v for d, v in acc.items()})
                if name in ("C", "C0"):
                    row["router_hits"] = router_hits(model)
                rows.append(row)
                print(f"{name:2s} n={n:5d} seed={seed} | d16 {acc[16]:.3f}"
                      f"{' | router ' + str(row['router_hits']) + '/10' if 'router_hits' in row else ''}",
                      flush=True)
    with open("strat2_results.json", "w") as f:
        json.dump(rows, f, indent=1)
    report(rows)
