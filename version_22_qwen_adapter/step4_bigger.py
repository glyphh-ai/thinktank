"""step4_bigger.py — the glyphh memory, made harder: 3,000 facts, shared
owners, five rewordings it was never taught.

step3's caveats, each turned into a test:

  shared owners   every person keeps TWO animals ("Piheba Doti keeps a hawk
                  named Nipira." and "... keeps a goat named Sabo."), and the
                  other one's name is always among the distractors, so the
                  invented person name alone can't pick the answer
  rewordings      taught only as "{p} keeps a {a} named", asked six ways:
                    taught   {p} keeps a {a} named
                    qa       Q: What is the name of {p}'s {a}?  A: Its name is
                    called   {p}'s {a} is called
                    belongs  The {a} that belongs to {p} goes by the name
                    ask      Ask {p} what their {a} is named, and they will say
                    answers  Among {p}'s animals, the {a} answers to
  scale           3,000 facts told once, a third corrected once (in context
                  stops being an option at this size: ~40k tokens of prompt)

Scored on a fixed sample of 300 facts per measure (all 3,000 are taught).
Arms: Qwen alone, + exact memory, + glyphh memory (step3's, tau set on dev
facts, seed 1).

Run:  python step4_bigger.py [--facts 3000] [--sample 300] [--arms exact,glyphh]
"""

import argparse
import json
import os
import time

import numpy as np

import memory
import qwen
import step3_glyphh as s3
from step1_adapter import Arm, HERE, K, book, candidates, invented, recall

ANIMALS = ["hawk", "goat", "hound", "mare", "owl", "ferret", "cat", "crow",
           "pony", "toad", "heron", "fox"]
ASK = {
    "taught": "{p} keeps a {a} named",
    "qa": "Q: What is the name of {p}'s {a}?\nA: Its name is",
    "called": "{p}'s {a} is called",
    "belongs": "The {a} that belongs to {p} goes by the name",
    "ask": "Ask {p} what their {a} is named, and they will say",
    "answers": "Among {p}'s animals, the {a} answers to",
}


def make_facts(n, seed=0):
    """n facts, two per person, each person's two animals different."""
    rng = np.random.default_rng(seed)
    people = n // 2
    first, last = invented(rng, 2 * people)[0::2], invented(rng, 2 * people)[1::2]
    pets = invented(np.random.default_rng(seed + 7), 2 * n)
    facts = []
    for i in range(people):
        p = f"{first[i]} {last[i]}"
        a, b = rng.choice(len(ANIMALS), 2, replace=False)
        for animal in (ANIMALS[a], ANIMALS[b]):
            k = len(facts)
            f = {key: t.format(p=p, a=animal) for key, t in ASK.items()}
            f.update(answer=f" {pets[2 * k]}.", new=f" {pets[2 * k + 1]}.", owner=i)
            facts.append(f)
    rng2 = np.random.default_rng(seed + 1)
    for i, f in enumerate(facts):
        sibling = i + 1 if i % 2 == 0 else i - 1          # same person, other animal
        others = [j for j in rng2.choice(len(facts), K + 2, replace=False) if j not in (i, sibling)]
        f["distract"] = [facts[sibling]["answer"]] + [facts[j]["answer"] for j in others[:K - 1]]
    return facts


def run_arm(arm, facts, pp, sample, phases):
    third = len(facts) // 3
    fixed, rest = facts[:third], facts[third:]
    rng = np.random.default_rng(3)
    s_all = [facts[i] for i in rng.choice(len(facts), sample, replace=False)]
    s_fix = [fixed[i] for i in rng.choice(len(fixed), sample, replace=False)]
    s_rest = [rest[i] for i in rng.choice(len(rest), sample, replace=False)]
    rows = []

    def measure(phase):
        t = time.time()
        row = {"arm": arm.name, "phase": phase}
        if phase.startswith("correct"):
            for key in ASK:
                nw, old = recall(arm.answer_bits(candidates(s_fix, key, "new", "answer")), True)
                row[f"new_{key}"], row[f"old_{key}"] = nw, old
            for key in ("taught", "qa"):
                row[f"untouched_{key}"], _ = recall(arm.answer_bits(candidates(s_rest, key, "answer")))
            say = ("new wins " + " ".join(f"{k} {row['new_' + k]:4.0%}" for k in ASK)
                   + f" | old wins (taught) {row['old_taught']:4.0%}"
                   + f" | untouched {row['untouched_taught']:4.0%}/{row['untouched_qa']:4.0%}")
        else:
            for key in ASK:
                row[key], _ = recall(arm.answer_bits(candidates(s_all, key, "answer")))
            say = "recall " + " ".join(f"{k} {row[k]:4.0%}" for k in ASK)
        row["pp_bpc"] = arm.damage(pp)
        rows.append(row)
        print(f"  {phase:<10} {say}  P&P {row['pp_bpc']:.3f} bpc  ({time.time() - t:.0f}s)", flush=True)

    print(arm.name, flush=True)
    if "before" in phases:
        measure("before")
    t = time.time()
    arm.teach([f["taught"] + f["answer"] for f in facts])
    print(f"  (taught {len(facts)} facts in {time.time() - t:.0f}s)", flush=True)
    measure("teach x1")
    arm.teach([f["taught"] + f["new"] for f in fixed])
    measure("correct x1")
    return rows


def calibrate(q, pct=99.5, n_dev=200, delta=0.0):
    """tau from a dev store (seed 1) of n_dev facts: pass the test size, so a
    bigger store (which unrelated text matches more often) gets a tighter tau."""
    dev = make_facts(n_dev, seed=1)
    m = s3.GlyphhMemory(q.tok, delta=delta)
    for f in dev:
        m.learn(q.encode(f["taught"] + f["answer"]))
    other = q.encode(book("frankenstein", 84)[50000:58000])
    _, s = m.nearest(m.contexts(other))
    tau = float(np.percentile(s, pct))
    print(f"  tau {tau:.3f} (unrelated text {pct}th pct, {n_dev} dev facts seed 1)", flush=True)
    return tau


def main(n, sample, arms, out, delta=0.0, dev_size=200, before=True):
    pp = book("pride_and_prejudice", 1342)[:20000]
    facts = make_facts(n)
    q = qwen.Qwen()
    print(f"{qwen.NAME}, {n} facts ({n // 2} people x 2 animals), scored on {sample} sampled\n"
          f"e.g. \"{facts[0]['taught']}{facts[0]['answer']}\" / \"{facts[1]['taught']}{facts[1]['answer']}\"\n")
    rows = []
    tau = None
    if "glyphh" in arms:
        print("calibrating")
        tau = calibrate(q, n_dev=dev_size or n, delta=delta)
        print()
    first = before
    for a in arms:
        if a == "exact":
            arm = Arm(q, "qwen + exact memory", exact=memory.ExactMemory())
        elif a == "glyphh":
            name = "qwen + glyphh memory" + (f" (mix delta {delta:g})" if delta else "")
            arm = Arm(q, name, exact=s3.GlyphhMemory(q.tok, tau, delta=delta))
        else:
            continue
        rows += run_arm(arm, facts, pp, sample, ("before",) if first else ())
        first = False
        print()
        with open(os.path.join(HERE, out), "w") as fh:
            json.dump(rows, fh, indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", type=int, default=3000)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--arms", default="glyphh,exact")
    ap.add_argument("--out", default="step4_results.json")
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--dev-size", type=int, default=200, help="0 = same as --facts")
    ap.add_argument("--no-before", action="store_true")
    a = ap.parse_args()
    main(a.facts, a.sample, a.arms.split(","), a.out, a.delta, a.dev_size, not a.no_before)
