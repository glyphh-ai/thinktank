"""
Report — results.json -> accuracy tables + the pre-registered gates.

Run:  python report.py
"""

import json

import numpy as np

from circuits import shortcut
from run import EVAL_DEPTHS, MODELS, SIZES, TRAIN_DEPTHS, eval_sets


def mean_acc(rows, model, n, depth):
    return float(np.mean([r["acc"][str(depth)] for r in rows if r["model"] == model and r["n"] == n]))


def id_acc(rows, model, n):
    return float(np.mean([mean_acc(rows, model, n, d) for d in TRAIN_DEPTHS]))


def main():
    rows = json.load(open("results.json"))

    evals = eval_sets()
    sc = {d: np.mean([shortcut(x.numpy(), o.numpy(), a.numpy()) == t
                      for x, o, a, t in zip(*evals[d])]) for d in EVAL_DEPTHS}

    print("\nmean accuracy over 3 seeds, by evaluation depth (trained on depths 1-4)")
    print("model     n  " + "".join(f"{d:>7d}" for d in EVAL_DEPTHS))
    print("shortcut    " + "".join(f"{sc[d]:7.3f}" for d in EVAL_DEPTHS) + "   <- last step only, no execution\n")
    for n in SIZES:
        for m in MODELS:
            print(f"{m:3s} {n:6d}  " + "".join(f"{mean_acc(rows, m, n, d):7.3f}" for d in EVAL_DEPTHS))
        print()

    hits = {n: [r["router_hits"] for r in rows if r["model"] == "C" and r["n"] == n] for n in SIZES}
    print("C router hits per seed (op -> its own tiny model, /10):",
          {n: h for n, h in hits.items()})

    c16, b16, a16, r16 = (mean_acc(rows, m, 8192, 16) for m in ("C", "B", "A", "C0"))
    gates = {
        "G1a loop generalizes: B and C >= 0.95 at d16, n=8192":
            (b16 >= 0.95 and c16 >= 0.95, f"B {b16:.3f}, C {c16:.3f}"),
        "G1b monolithic does not: A <= 0.75 at d16, n=8192":
            (a16 <= 0.75, f"A {a16:.3f}"),
        "G2 CORE tiny models buy data: C - B >= 0.10 at d16, n=256":
            (mean_acc(rows, "C", 256, 16) - mean_acc(rows, "B", 256, 16) >= 0.10,
             f"C {mean_acc(rows, 'C', 256, 16):.3f} - B {mean_acc(rows, 'B', 256, 16):.3f}"),
        "G3 router is readable: >= 9/10 ops correct in every seed, n=8192":
            (min(hits[8192]) >= 9, f"{hits[8192]}"),
        "G4 no in-distribution cost: C ID >= B ID - 0.01, n=8192":
            (id_acc(rows, "C", 8192) >= id_acc(rows, "B", 8192) - 0.01,
             f"C {id_acc(rows, 'C', 8192):.3f}, B {id_acc(rows, 'B', 8192):.3f}"),
        "G5 content not shape: C - C0 >= 0.10 at d16, n=256":
            (mean_acc(rows, "C", 256, 16) - mean_acc(rows, "C0", 256, 16) >= 0.10,
             f"C {mean_acc(rows, 'C', 256, 16):.3f} - C0 {mean_acc(rows, 'C0', 256, 16):.3f}"),
    }
    print("\nGATES")
    for name, (ok, detail) in gates.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}   [{detail}]")


if __name__ == "__main__":
    main()
