"""baselines.py — the no-training rules on each test set, by move.

Run: python baselines.py
"""

import collections
import json
import os

import world as W

HERE = os.path.dirname(os.path.abspath(__file__))


def precedents(s):
    out = []
    for i, d, c, dec, held in s["got"]:
        p = W.Precedent(d, c, dec, held)
        p.id = i
        out.append(p)
    return out


def main():
    for name in ("unseen", "harder"):
        rows = json.load(open(f"{HERE}/data/test_{name}.json"))
        hit, n = collections.Counter(), collections.Counter()
        for s in rows:
            ps = precedents(s)
            near = W.nearest(ps, s["case"])
            guess = {"nearest": near.decision, "nearest+outcome": near.effective(s["labels"]),
                     "latest+outcome": ps[-1].effective(s["labels"])}
            n[s["kind"]] += 1
            for rule, d in guess.items():
                hit[rule, s["kind"]] += d == s["decision"]
                hit[rule, "all"] += d == s["decision"]
        print(f"\n{name} (n={len(rows)})")
        for rule in ("nearest", "nearest+outcome", "latest+outcome"):
            cells = " ".join(f"{k}={hit[rule, k] / n[k]:.0%}" for k in sorted(n))
            print(f"  {rule:16s} all={hit[rule, 'all'] / len(rows):.0%}  {cells}")


if __name__ == "__main__":
    main()
