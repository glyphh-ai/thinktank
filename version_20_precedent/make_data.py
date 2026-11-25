"""make_data.py — balanced train and test sets, one folder per arm.

Run: python make_data.py [--train 6000 --test 450]
"""

import argparse
import json
import os
import random
import time

import world as W

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = ("apply", "distinguish", "reverse", "overrule")


def build(n, split, seed, **kw):
    rng = random.Random(seed)
    out, t, tries = [], time.time(), 0
    for kind in KINDS:
        got = 0
        while got < n // 4:
            tries += 1
            s = W.sample(rng, split, want_kind=kind, **kw)
            if s:
                out.append(s)
                got += 1
    rng.shuffle(out)
    print(f"{split} {kw}: {len(out)} kept of {tries} tried, {time.time() - t:.0f}s", flush=True)
    return out


def target(s, arm):
    d = f"decision: {s['decision']}"
    return {"decision": f" {d}", "move": f" {s['move']}; {d}", "after": f" {d}; {s['move']}"}[arm]


def main(a):
    os.makedirs(f"{HERE}/data", exist_ok=True)
    train = build(a.train, "train", 1)
    valid = build(160, "train", 2)
    for arm in ("decision", "move", "after"):
        os.makedirs(f"{HERE}/data/{arm}", exist_ok=True)
        for name, rows in (("train", train), ("valid", valid)):
            with open(f"{HERE}/data/{arm}/{name}.jsonl", "w") as f:
                for s in rows:
                    f.write(json.dumps({"prompt": s["prompt"], "completion": target(s, arm)}) + "\n")
    for name, kw in (("unseen", {}), ("harder", dict(n_attr=5, k=14))):
        rows = build(a.test, "test", 3, **kw)
        json.dump(rows, open(f"{HERE}/data/test_{name}.json", "w"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=6000)
    p.add_argument("--test", type=int, default=450)
    main(p.parse_args())
