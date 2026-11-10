"""step12_unstructured.py — how much free text can be pumped in as facts, and how does it do?

Every fact so far was one clean sentence in one template. A customer's data
is not: it is case notes, a paragraph each, several facts in running prose,
filler in between, and after the first sentence the customer is "they".
Here the memory is taught such notes whole (every word of them, as it would
be in production: nobody marks which words are the facts) and asked about
them in wordings the notes never use.

  a note      an opening that names the customer, four facts (account
              manager, town, product, discount code; invented values), two
              filler sentences, the facts in random order and phrasing
  near        every fact sentence names the customer again
              ("Piheba Doti's account manager is Nipira.")
  far         only the opening names them; the facts say "they"
              ("Their account manager is Nipira."), so the name is outside
              the memory's 8-word window when the value is written
  asked       two ways per fact, neither used in a note: a question
              ("Q: Who is the account manager for {p}? A: The account
              manager is") and a rewording; right value vs 20 other cases'

Two memories, step4c's settings (delta 0.05, function words out of the key):
  window      step3/4's key: the last 8 content words
  carry       the same + the invented names met earlier in the text (up to
              400 characters back), at half voice: who the text is about
              stays in the key after the window has moved on

Scale: 250, 1,000 and 2,000 notes (1,000 to 8,000 facts). tau per store
size and kind, from a dev store of the same size (seed 1) against
Frankenstein, 99.5%. At the largest size a third of the cases get an update
note ("... account manager is now X") and the manager is asked again.

Run:  QWEN=Qwen3-4B-4bit python step12_unstructured.py
"""

import argparse
import json
import os
import time

import numpy as np

import qwen
import step3_glyphh as s3
from step1_adapter import Arm, HERE, K, book, invented, recall

s3.STOP |= s3.FUNCTION
ATTRS = ["manager", "town", "product", "code"]
INTRO = ["{p} wrote in on Tuesday about a late delivery.",
         "{p} called this morning with a billing question.",
         "A message came in from {p} over the weekend.",
         "{p} reached out through chat regarding a damaged parcel."]
FILL = ["The call lasted about ten minutes.", "No refund was requested this time.",
        "A follow-up was promised within the week.", "The line dropped once and was reconnected.",
        "Everything else on the account looked normal.", "The earlier ticket had already been closed.",
        "A survey link went out afterwards.", "Notes from the previous agent were thin."]
NEAR = {"manager": ["{p}'s account manager is {v}.", "The account manager for {p} is {v}."],
        "town": ["{p} lives in {v}.", "{p} is currently living in the town of {v}."],
        "product": ["{p} bought the {v} last spring.", "The product {p} owns is the {v}."],
        "code": ["{p} was given the discount code {v}.", "The discount code issued to {p} was {v}."]}
FAR = {"manager": ["Their account manager is {v}.", "The account manager on file is {v}."],
       "town": ["They live in {v}.", "Home is the town of {v}."],
       "product": ["They bought the {v} last spring.", "The product in question is the {v}."],
       "code": ["They were given the discount code {v}.", "The discount code issued was {v}."]}
ASK = {"manager": ["Q: Who is the account manager for {p}? A: The account manager is",
                   "The person who manages {p}'s account is"],
       "town": ["Q: Which town does {p} live in? A: The town is", "{p}'s home town is"],
       "product": ["Q: Which product did {p} buy? A: The product is the", "The item purchased by {p} was the"],
       "code": ["Q: What discount code did {p} get? A: The discount code is", "{p} can use the discount code"]}
UPDATE = {"near": "Update on {p}: {p}'s account manager is now {v}.",
          "far": "An update came in for {p}. Their account manager is now {v}."}
CARRY = 0.5


class Carry(s3.GlyphhMemory):
    def context_of(self, text):
        v = super().context_of(text)
        low = text[-400:].lower()
        found = s3._WORD.findall(low)
        if found and low[-1:].isalpha():
            found.pop()
        earlier = [w for w in found if w not in s3.STOP][:-s3.WINDOW]
        names = {w for w in earlier if s3.LEXICON.entry(w) is None and s3.LEXICON.lemma(w) is None}
        if not names or not v.any():
            return v
        u = sum(self._wglyph(w) for w in names)
        v = v + CARRY * u / (np.linalg.norm(u) or 1.0)
        return v / np.linalg.norm(v)


def make_cases(n, seed=0):
    rng = np.random.default_rng(seed)
    words = invented(rng, 7 * n)
    cases = []
    for i in range(n):
        w = words[7 * i:7 * i + 7]
        cases.append({"p": f"{w[0]} {w[1]}", **dict(zip(ATTRS, w[2:6])), "new": w[6]})
    return cases


def note(c, kind, rng):
    forms = NEAR if kind == "near" else FAR
    body = [forms[a][rng.integers(2)].format(p=c["p"], v=c[a]) for a in rng.permutation(ATTRS)]
    for f in rng.choice(len(FILL), 2, replace=False):
        body.insert(int(rng.integers(len(body) + 1)), FILL[f])
    return " ".join([INTRO[rng.integers(len(INTRO))].format(p=c["p"])] + body)


def groups(cases, sample, attr, form, rng, key=None, extra=None):
    """(prompt, [right, 20 other cases' values of the same attribute (, the old value)])."""
    out = []
    for i in sample:
        c = cases[i]
        others = [cases[j][attr] for j in rng.choice(len(cases), K + 1, replace=False) if j != i][:K]
        cands = [f" {c[key or attr]}."] + [f" {o}." for o in others] + ([f" {c[extra]}."] if extra else [])
        out.append((ASK[attr][form].format(p=c["p"]), cands))
    return out


def tau_for(q, cls, n, kind, delta, pct=99.5):
    rng = np.random.default_rng(101)
    m = cls(q.tok, delta=delta)
    for c in make_cases(n, seed=1):
        m.learn(q.encode(note(c, kind, rng)))
    _, s = m.nearest(m.contexts(q.encode(book("frankenstein", 84)[50000:58000])))
    return float(np.percentile(s, pct))


def main(scales, sample, out, delta=0.05):
    q = qwen.Qwen()
    pp = book("pride_and_prejudice", 1342)[:20000]
    base = Arm(q, "alone").damage(pp)
    print(f"{qwen.NAME}; notes of four facts each, asked two untaught ways per fact, {sample} cases sampled; "
          f"English alone {base:.3f} bpc")
    c0 = make_cases(1)[0]
    for kind in ("near", "far"):
        print(f"  {kind}: {note(c0, kind, np.random.default_rng(5))}")
    print()
    results = {"base_bpc": base, "rows": []}
    for n in scales:
        cases = make_cases(n)
        pick = np.random.default_rng(3).choice(n, min(sample, n), replace=False)
        for kind in ("near", "far"):
            for name, cls in (("window", s3.GlyphhMemory), ("carry", Carry)):
                t = time.time()
                mem = cls(q.tok, delta=delta)
                mem.tau = tau_for(q, cls, n, kind, delta)
                arm = Arm(q, name, exact=mem)
                rng = np.random.default_rng(11)
                texts = [note(c, kind, rng) for c in cases]
                t1 = time.time()
                arm.teach(texts)
                teach = time.time() - t1
                row = {"notes": n, "facts": 4 * n, "kind": kind, "memory": name, "tau": mem.tau,
                       "keys": len(mem.counts), "store_mb": len(mem.counts) * s3.DIM * 4 / 1e6,
                       "words": sum(len(x.split()) for x in texts), "teach_s": teach}
                r2 = np.random.default_rng(17)
                for a in ATTRS:
                    for form, tag in enumerate(("qa", "reworded")):
                        row[f"{a}_{tag}"], _ = recall(arm.answer_bits(groups(cases, pick, a, form, r2)))
                row["recall"] = float(np.mean([row[f"{a}_{tag}"] for a in ATTRS for tag in ("qa", "reworded")]))
                say = ""
                if n == scales[-1]:
                    row["bpc"] = arm.damage(pp)
                    third = n // 3
                    arm.teach([UPDATE[kind].format(p=c["p"], v=c["new"]) for c in cases[:third]])
                    fixed = np.random.default_rng(4).choice(third, min(sample, third), replace=False)
                    rest = third + np.random.default_rng(5).choice(n - third, min(sample, n - third), replace=False)
                    for form, tag in enumerate(("qa", "reworded")):
                        nw, old = recall(arm.answer_bits(groups(cases, fixed, "manager", form, r2, "new", "manager")), True)
                        row[f"new_{tag}"], row[f"old_{tag}"] = nw, old
                        row[f"untouched_{tag}"], _ = recall(arm.answer_bits(groups(cases, rest, "manager", form, r2)))
                    say = (f" | English {row['bpc']:.3f} | corrected: new {row['new_qa']:.0%}/{row['new_reworded']:.0%}"
                           f" old {row['old_qa']:.0%}/{row['old_reworded']:.0%}"
                           f" untouched {row['untouched_qa']:.0%}/{row['untouched_reworded']:.0%}")
                print(f"{n:>5} notes {kind:<4} {name:<6} recall {row['recall']:4.0%}  ("
                      + " ".join(f"{a} {row[a + '_qa']:.0%}/{row[a + '_reworded']:.0%}" for a in ATTRS)
                      + f")  tau {mem.tau:.2f}  {row['words']:,} words -> {row['keys']:,} keys, "
                      f"{row['store_mb']:.0f} MB, taught in {teach:.0f}s{say}  ({time.time() - t:.0f}s)", flush=True)
                results["rows"].append(row)
                with open(os.path.join(HERE, out), "w") as fh:
                    json.dump(results, fh, indent=1, default=float)
                del mem, arm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="250,1000,2000")
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--out", default="step12_results.json")
    a = ap.parse_args()
    main([int(x) for x in a.scales.split(",")], a.sample, a.out)
