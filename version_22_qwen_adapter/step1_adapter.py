"""step1_adapter.py — a counter memory beside Qwen3-1.7B, against LoRA and the prompt.

Same 300 invented facts as version_21 ("Piheba Doti keeps a hawk named
Nipira."), now on a real LLM. Each arm is told every fact (1-2 passes),
then a third get a new name (1-2 passes). After every phase:

  recall      right name beats 20 others, asked the way it was taught
  paraphrase  the same, asked "Q: What is the name of Piheba Doti's hawk?
              A: Its name is"
  corrected   new name beats the old one and the 20 others
  untouched   recall on the facts that were never corrected
  damage      Qwen bits/char on 20k chars of Pride and Prejudice

Arms:
  qwen                  no learning
  + exact memory        counters keyed on the last 4/8/12 tokens
  + angle memory        counters keyed on Qwen's hidden-state sign bits
  + both
  LoRA lr=..            rank 8 on the last 16 layers, one Adam step per fact
  in context            every fact told so far sits in the prompt (the
                        strong baseline while it fits)

The angle memory's layer and tau are chosen on a separate set of dev facts
(seed 1), never on the test facts.

Run:  python step1_adapter.py [--facts 300] [--passes 2] [--arms ...]
"""

import argparse
import json
import math
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm.tuner.utils import linear_to_lora_layers

import memory
import qwen

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "version_19_glyph"))
from step2_codec import book  # noqa: E402

ANIMALS = ["hawk", "goat", "hound", "mare", "owl", "ferret", "cat", "crow",
           "pony", "toad", "heron", "fox"]
K = 20
MARGIN = 64


def invented(rng, n):
    cons, vow = "bdfghklmnprstvz", "aeiou"
    out, seen = [], set()
    while len(out) < n:
        w = "".join(rng.choice(list(cons)) + rng.choice(list(vow))
                    for _ in range(rng.integers(2, 4))).capitalize()
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def make_facts(n, seed=0):
    rng = np.random.default_rng(seed)
    first, last, pet, pet2 = (invented(rng, 4 * n)[k::4] for k in range(4))
    facts = []
    for i in range(n):
        a = ANIMALS[i % len(ANIMALS)]
        facts.append({
            "taught": f"{first[i]} {last[i]} keeps a {a} named",
            "para": f"Q: What is the name of {first[i]} {last[i]}'s {a}?\nA: Its name is",
            "answer": f" {pet[i]}.", "new": f" {pet2[i]}."})
    rng2 = np.random.default_rng(seed + 1)
    for i, f in enumerate(facts):
        others = [j for j in rng2.choice(n, K + 1, replace=False) if j != i][:K]
        f["distract"] = [facts[j]["answer"] for j in others]
    return facts


def sentence(f, key):
    return f["taught"] + f[key]


# ------------------------------------------------------------------ scoring

def candidates(facts, prompt_key, target_key, extra=None):
    groups = []
    for f in facts:
        c = [f[target_key]] + [d for d in f["distract"] if d != f[target_key]][:K]
        if extra:
            c.append(f[extra])
        groups.append((f[prompt_key], c))
    return groups


def recall(bits, extra=False):
    win = bits.argmin(1)
    return (win == 0).mean(), ((win == bits.shape[1] - 1).mean() if extra else None)


class Arm:
    """Base: Qwen, optionally with memories. Scores answer bits per candidate."""

    def __init__(self, q, name, exact=None, angle=None):
        self.q, self.name, self.exact, self.angle = q, name, exact, angle

    def layers(self):
        return (self.angle.layer,) if self.angle else ()

    def answer_bits(self, groups, prefix=None):
        seqs, spans = [], []
        for prompt, cands in groups:
            p = self.q.encode(prompt)
            for c in cands:
                s = self.q.encode(prompt + c)
                seqs.append(s)
                spans.append(len(p))
        out = self.q.run(seqs, self.layers(), batch=6 if prefix else 48, prefix_cache=prefix)
        bits = []
        for s, a, (lp, hid) in zip(seqs, spans, out):
            if self.exact:
                lp = self.exact.refine(s, lp, start=a - 1)
            if self.angle:
                lp = self.angle.refine(hid[self.angle.layer], s, lp, start=a - 1)
            bits.append(-lp[a - 1:].sum() / math.log(2))
        n = len(groups[0][1])
        return np.array(bits).reshape(len(groups), n)

    def teach(self, texts):
        if not (self.exact or self.angle):
            return
        seqs = [self.q.encode(t) for t in texts]
        out = self.q.run(seqs, self.layers(), batch=48) if self.angle else [None] * len(seqs)
        for s, o in zip(seqs, out):
            if self.exact:
                self.exact.learn(s)
            if self.angle:
                self.angle.learn(o[1][self.angle.layer], s)

    def damage(self, text):
        toks = self.q.encode(text)
        wins = [toks[k:k + 1024] for k in range(0, len(toks), 1024)]
        total = 0.0
        for s, (lp, hid) in zip(wins, self.q.run(wins, self.layers(), batch=4)):
            if self.exact:
                lp = self.exact.refine(s, lp)
            if self.angle:
                lp = self.angle.refine(hid[self.angle.layer], s, lp)
            total += -lp.sum() / math.log(2)
        return total / len(text)


class LoRAArm(Arm):
    def __init__(self, q, lr):
        super().__init__(q, f"LoRA lr={lr:g}")
        q.model.freeze()
        linear_to_lora_layers(q.model, 16, {"rank": 8, "scale": 20.0, "dropout": 0.0})
        self.opt = optim.Adam(lr)

        def loss(model, x, y):
            return nn.losses.cross_entropy(model(x), y, reduction="mean")
        self.step = nn.value_and_grad(q.model, loss)

    def teach(self, texts):
        for t in texts:
            x = np.array(self.q.encode(t), np.int32)[None]
            _, g = self.step(self.q.model, mx.array(x[:, :-1]), mx.array(x[:, 1:]))
            self.opt.update(self.q.model, g)
            mx.eval(self.q.model.trainable_parameters(), self.opt.state)


class InContextArm(Arm):
    def __init__(self, q):
        super().__init__(q, "in context")
        self.told = []

    def teach(self, texts):
        self.told += texts

    def answer_bits(self, groups, prefix=None):
        if not self.told:
            return super().answer_bits(groups)
        cache = qwen.prefix_cache(self.q, self.q.encode("\n".join(self.told) + "\n\n"))
        return super().answer_bits(groups, prefix=cache)

    def damage(self, text):
        return float("nan")          # its cost is context, not weights


# ------------------------------------------------------------------ the run

def run_arm(arm, facts, pp, passes):
    third = len(facts) // 3
    fixed, rest = facts[:third], facts[third:]
    rows = []

    def measure(phase):
        row = {"arm": arm.name, "phase": phase}
        t = time.time()
        if phase.startswith("correct"):
            nw, old = recall(arm.answer_bits(candidates(fixed, "taught", "new", "answer")), True)
            npw, _ = recall(arm.answer_bits(candidates(fixed, "para", "new", "answer")), True)
            keep, _ = recall(arm.answer_bits(candidates(rest, "taught", "answer")))
            row.update(new_wins=nw, old_wins=old, new_wins_para=npw, untouched=keep)
            say = f"new wins {nw:4.0%} (para {npw:4.0%}), old wins {old:4.0%}, untouched {keep:4.0%}"
        else:
            r, _ = recall(arm.answer_bits(candidates(facts, "taught", "answer")))
            rp, _ = recall(arm.answer_bits(candidates(facts, "para", "answer")))
            row.update(recall=r, recall_para=rp)
            say = f"recall {r:4.0%}  paraphrase {rp:4.0%}"
        row["pp_bpc"] = arm.damage(pp)
        rows.append(row)
        print(f"  {phase:<10} {say:<62} P&P {row['pp_bpc']:.3f} bpc  ({time.time() - t:.0f}s)",
              flush=True)

    print(arm.name, flush=True)
    measure("before")
    for k in range(passes):
        arm.teach([sentence(f, "answer") for f in facts])
        measure(f"teach x{k + 1}")
    for k in range(passes):
        arm.teach([sentence(f, "new") for f in fixed])
        measure(f"correct x{k + 1}")
    return rows


def calibrate(q, layers, n_dev=100, pct=5):
    """Pick the angle layer and tau on dev facts (seed 1) + unrelated text.

    mu = mean hidden state over unrelated text (Frankenstein). tau = the
    `pct` percentile of the nearest-entry distance from unrelated text to
    the dev memory, so unrelated text rarely fires it. The layer is the one
    with the best dev paraphrase recall. (A wider dev sweep, layers 4-21,
    tau at the 0.5th/5th percentile, margin 0/16/64, picked layer 10, 5th,
    margin 64: dev recall 100%, paraphrase 18%.)
    """
    dev = make_facts(n_dev, seed=1)
    other = book("frankenstein", 84)[20000:32000]
    toks = q.encode(other)
    wins = [toks[k:k + 512] for k in range(0, len(toks), 512)]
    hs = q.run(wins, layers, batch=4)
    best = None
    for layer in layers:
        H = np.concatenate([h[layer] for _, h in hs])
        mu = H.mean(0)
        mem = memory.AngleMemory(layer, mu, tau=0, margin=MARGIN)
        arm = Arm(q, "dev", angle=mem)
        arm.teach([sentence(f, "answer") for f in dev])
        near = mem.dist(memory.codes(H, mu)).min(1)
        tau = int(np.percentile(near, pct))
        mem.tau = tau
        rp, _ = recall(arm.answer_bits(candidates(dev, "para", "answer")))
        r, _ = recall(arm.answer_bits(candidates(dev, "taught", "answer")))
        print(f"  dev layer {layer:>5}: tau {tau:4d} of {mu.shape[0]} bits, "
              f"recall {r:4.0%}, paraphrase {rp:4.0%}", flush=True)
        if best is None or rp > best[0] or (rp == best[0] and r > best[1]):
            best = (rp, r, layer, mu, tau)
    return best[2:]


def main(n, passes, arms, lrs, out="step1_results.json"):
    pp = book("pride_and_prejudice", 1342)[:20000]
    facts = make_facts(n)
    q = qwen.Qwen()
    print(f"{qwen.NAME}, {n} invented facts, e.g. \"{sentence(facts[0], 'answer')}\"\n"
          f"paraphrase: \"{facts[0]['para']}{facts[0]['answer']}\"\n")
    rows = []
    if any(a in arms for a in ("angle", "both")):
        print("calibrating the angle memory on dev facts")
        layer, mu, tau = calibrate(q, (10,))
        print(f"  -> layer {layer}, tau {tau}\n")
    for a in arms:
        if a == "qwen":
            arm = Arm(q, "qwen alone")
        elif a == "exact":
            arm = Arm(q, "qwen + exact memory", exact=memory.ExactMemory())
        elif a == "angle":
            arm = Arm(q, "qwen + angle memory",
                      angle=memory.AngleMemory(layer, mu, tau, margin=MARGIN))
        elif a == "both":
            arm = Arm(q, "qwen + exact + angle", exact=memory.ExactMemory(),
                      angle=memory.AngleMemory(layer, mu, tau, margin=MARGIN))
        elif a == "context":
            arm = InContextArm(q)
        else:
            continue
        rows += run_arm(arm, make_facts(n), pp, passes)
        print()
        save(rows, out)
    for lr in lrs if "lora" in arms else []:
        rows += run_arm(LoRAArm(qwen.Qwen(), lr), make_facts(n), pp, passes)
        print()
        save(rows, out)


def save(rows, out):
    with open(os.path.join(HERE, out), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", type=int, default=300)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--arms", default="qwen,exact,angle,both,context,lora")
    ap.add_argument("--lrs", default="1e-5,1e-4,1e-3")
    ap.add_argument("--out", default="step1_results.json")
    a = ap.parse_args()
    main(a.facts, a.passes, a.arms.split(","), [float(x) for x in a.lrs.split(",")], a.out)
