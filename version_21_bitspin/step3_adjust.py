"""step3_adjust.py — learn new facts from the forward pass, and what it breaks.

The claim under test: a model whose weights are counters can take in a new
fact in one forward pass, and a later correction moves only that fact,
while a regular transformer has to push shared weights and pays for it
elsewhere.

Facts are invented so neither model can know them:
    "Orlek Vashti keeps a hawk named Tumeri."
Recall = the true name scores best among itself and 20 other facts' names.

Phases, each followed by recall on every fact + bits/char on 50k chars of
Pride and Prejudice (frozen, to measure damage):
  0  before        base models from step1 / step2
  1  teach         every fact once, in order
  2  correct       the first third get a new name, shown once
                   (new name must win, old must lose, the rest must hold)

Bitspin learns by running forward (learn=True). The transformer gets its
fairest online version: `--steps` Adam steps per fact at learning rate `--lr`,
swept.

Run:  python step3_adjust.py [--facts 300] [--lrs 1e-4,3e-4,1e-3] [--steps 1,3]
"""

import argparse
import json
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import bitspin
import step1_lm
import step2_transformer as st

HERE = os.path.dirname(os.path.abspath(__file__))
ANIMALS = ["hawk", "goat", "hound", "mare", "owl", "ferret", "cat", "crow",
           "pony", "toad", "heron", "fox"]
K_DISTRACT = 20


def invented(rng, n):
    cons, vow = "bdfghklmnprstvz", "aeiou"
    out, seen = [], set()
    while len(out) < n:
        w = "".join(rng.choice(list(cons)) + rng.choice(list(vow))
                    for _ in range(rng.integers(2, 4)))
        w = w.capitalize()
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def make_facts(n, seed=0):
    rng = np.random.default_rng(seed)
    first, last, pet, pet2 = (invented(rng, 4 * n)[k::4] for k in range(4))
    facts = []
    for i in range(n):
        p = f"{first[i]} {last[i]} keeps a {ANIMALS[i % len(ANIMALS)]} named "
        facts.append({"prompt": p, "answer": f"{pet[i]}.", "new": f"{pet2[i]}."})
    rng2 = np.random.default_rng(seed + 1)
    for i, f in enumerate(facts):
        others = [j for j in rng2.choice(n, K_DISTRACT + 1, replace=False) if j != i][:K_DISTRACT]
        f["distract"] = [facts[j]["answer"] for j in others]
    return facts


# ------------------------------------------------------------------ scoring

def pairs_for(facts, key, extra=None):
    """(prompt, candidate) pairs: the target first, then distractors (+ extra)."""
    out = []
    for f in facts:
        cands = [f[key]] + f["distract"] + ([f[extra]] if extra else [])
        out.append([(f["prompt"], c) for c in cands])
    return out


def bs_bits(m, pairs):
    flat = [pc for group in pairs for pc in group]
    text, spans, pos = b"", [], 0
    for p, a in flat:
        chunk = ("\n" + p + a).encode()
        spans.append((pos + 1 + len(p), pos + len(chunk)))
        text += chunk
        pos += len(chunk)
    m.run(text, learn=False, batch=8192, tail=b"\n\n")
    per = m.last_per_byte
    bits = np.array([per[a:b].sum() for a, b in spans])
    return bits.reshape(len(pairs), -1)


def tf_bits(model, pairs):
    flat = [pc for group in pairs for pc in group]
    seqs = [np.frombuffer(("\n" + p + a).encode(), np.uint8) for p, a in flat]
    L = max(map(len, seqs))
    out = np.zeros(len(flat))
    model.eval()
    for k in range(0, len(seqs), 256):
        chunk = seqs[k:k + 256]
        x = np.zeros((len(chunk), L), np.int32)
        for r, s in enumerate(chunk):
            x[r, :len(s)] = s
        logp = nn.log_softmax(model(mx.array(x[:, :-1])), axis=-1)
        nll = -np.array(mx.take_along_axis(logp, mx.array(x[:, 1:])[..., None], axis=-1)[..., 0])
        for r, (s, (p, a)) in enumerate(zip(chunk, flat[k:k + 256])):
            start = 1 + len(p.encode())              # first answer byte index in s
            out[k + r] = nll[r, start - 1:len(s) - 1].sum() / math.log(2)
    model.train()
    return out.reshape(len(pairs), -1)


def report(bits, extra_old=False):
    """recall = target best; with extra_old, the last column is the old answer."""
    acc = (bits.argmin(1) == 0).mean()
    ans = bits[:, 0].mean()
    old = (bits.argmin(1) == bits.shape[1] - 1).mean() if extra_old else None
    return acc, ans, old


# ------------------------------------------------------------------ the two models

class BS:
    """local=True: after training only counters move (the shared mix is frozen).
    local=False: the shared mix keeps learning too."""

    def __init__(self, train, local):
        self.name = "bitspin, counters only" if local else "bitspin, counters + shared mix"
        self.m = bitspin.BitSpin()
        self.m.run(train)
        self.m.mix_learns = not local

    def bits(self, pairs):
        return bs_bits(self.m, pairs)

    def teach(self, texts):
        for s in texts:                               # one forward pass per fact
            self.m.run(("\n" + s).encode(), learn=True, batch=4096, tail=b"\n\n")

    def damage(self, pp):
        return self.m.run(pp, learn=False, tail=b"\n\n") / len(pp)


class TF:
    def __init__(self, lr, steps):
        self.name = f"transformer lr={lr:g} x{steps}"
        self.model = st.ByteLM()
        self.model.load_weights(os.path.join(st.RUNS, "byte_lm_d384_L6.safetensors"))
        self.opt = optim.Adam(lr)
        self.steps = steps
        self.step = nn.value_and_grad(self.model, st.loss_fn)

    def bits(self, pairs):
        return tf_bits(self.model, pairs)

    def teach(self, texts):
        self.model.eval()                             # no dropout while learning a fact
        for s in texts:
            x = np.frombuffer(("\n" + s).encode(), np.uint8).astype(np.int32)[None]
            for _ in range(self.steps):
                _, g = self.step(self.model, mx.array(x[:, :-1]), mx.array(x[:, 1:]))
                self.opt.update(self.model, g)
                mx.eval(self.model.parameters(), self.opt.state)
        self.model.train()

    def damage(self, pp):
        return st.bits_per_char(self.model, pp)


def run_arm(arm, facts, pp, passes):
    third = len(facts) // 3
    fixed, rest = facts[:third], facts[third:]
    for f in fixed:
        f["old"] = f["answer"]
    rows = []

    def measure(phase):
        row = {"model": arm.name, "phase": phase, "pp_bpc": arm.damage(pp)}
        if phase.startswith("correct"):
            na, nb, still_old = report(arm.bits(pairs_for(fixed, "new", extra="old")), True)
            keep, _, _ = report(arm.bits(pairs_for(rest, "answer")))
            row.update(new_wins=na, old_wins=still_old, untouched_recall=keep, answer_bits=nb)
            say = (f"new name wins {na:4.0%}, old still wins {still_old:4.0%}, "
                   f"untouched recall {keep:4.0%}")
        else:
            a, b, _ = report(arm.bits(pairs_for(facts, "answer")))
            row.update(recall=a, answer_bits=b)
            say = f"recall {a:4.0%}  answer {b:5.1f} bits"
        rows.append(row)
        print(f"  {phase:<10} {say:<62} P&P {row['pp_bpc']:.3f} bpc", flush=True)

    print(arm.name, flush=True)
    measure("before")
    for k in range(passes):
        arm.teach([f["prompt"] + f["answer"] for f in facts])
        measure(f"teach x{k + 1}")
    for k in range(passes):
        arm.teach([f["prompt"] + f["new"] for f in fixed])
        measure(f"correct x{k + 1}")
    return rows


def main(n, lrs, steps, passes, only, tag=""):
    facts = make_facts(n)
    pp = step1_lm.test_text(50_000)
    print(f"{n} invented facts, e.g. \"{facts[0]['prompt']}{facts[0]['answer']}\"; "
          f"recall = right name beats {K_DISTRACT} others; a third are corrected later\n")
    rows = []
    if only in ("", "bitspin"):
        for local in (True, False):
            t = time.time()
            rows += run_arm(BS(step1_lm.corpus(), local), make_facts(n), pp, passes)
            print(f"  ({time.time() - t:.0f}s)\n")
    if only in ("", "transformer"):
        for lr in lrs:
            for k in steps:
                t = time.time()
                rows += run_arm(TF(lr, k), make_facts(n), pp, passes)
                print(f"  ({time.time() - t:.0f}s)\n")
    with open(os.path.join(HERE, f"step3_{only or 'all'}{tag}.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", type=int, default=300)
    ap.add_argument("--lrs", default="1e-4,3e-4,1e-3")
    ap.add_argument("--steps", default="1,3")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--only", default="", choices=["", "bitspin", "transformer"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a.facts, [float(x) for x in a.lrs.split(",")], [int(x) for x in a.steps.split(",")],
         a.passes, a.only, a.tag)
