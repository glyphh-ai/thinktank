"""step11_draft_gate.py — a gate on the whole predicted reply, for rules that act on a topic.

step10's gates sit on a spot of the draft (the next 3 tokens + the last 3).
That is right for a rule that acts at one kind of spot (the sign-off at the
end: 100%) and wrong for a rule that acts on a topic: the ban's moments
scatter over many tokens, and every earlier spot of the same draft counted
as a negative for it although the topic was already there. The ban gate
fired on 15% of the requests for a model.

Here each rule also gets a DRAFT gate, asked once per predicted reply:
does this draft (its gist: the content words of the question and the
predicted reply) look like the practice drafts on which the rule went on to
act, or like those it left alone? If it does, the reply halts before its
first word, the rule comes into view and the reply is drafted again; the
draft was only a prediction, so nothing is lost.

  positives   practice drafts written without the rule on which it acted
  negatives   practice drafts written without the rule on which it never did
  a rule with fewer than 10 of either gets no draft gate: the sign-off acts
  on every reply, so it has no negatives and stays a spot gate (step10's)

Rehearsal, practice, recall at a halt and the spot gates are step10's
(gist key, one gate per (rule, token), taus from the quiet spots).

Four variants are compared on DEV (the 80 questions already used: step5's
and step9's) and one is picked before the test run by step10's score,
sign-off + ban - (ban rule in view on general questions):
  none       step10 as it was
  bundle     key = the positives bundled;            tau = negatives' 99.5%
  contrast   key = positives minus negatives;        tau = negatives' 99.5%
  midpoint   key = positives minus negatives;        tau = halfway between the two means
Then one run on 40 questions never used (FOURTH_*).

Run:  python step11_draft_gate.py
"""

import argparse
import json
import os
import time

import numpy as np

import qwen
import step7_rehearse as s7
import step8_lookback as s8
import step9_recall as s9
import step10_gates as s10
from step1_adapter import HERE
from step5_instructions import GENERAL, ML
from step10_gates import BAN, R, line, run, unit

FOURTH_GENERAL = [
    "How do I patch a bike tire?", "Why is grass green?", "How do I fold a fitted sheet?",
    "What is a good first board game?", "How can I drink more water?", "Why do cats knead?",
    "How do I clean a mirror without streaks?", "What is the capital of Brazil?",
    "How do I get better at swimming?", "Why does salt melt ice?",
    "How often should I back up my phone?", "What makes a good toast at a wedding?",
    "How do I get wax out of a carpet?", "Why do we hiccup?", "How do I start a vegetable garden?",
    "What is a credit score?", "How can I type faster?", "Why does the moon change shape?",
    "How do I make hummus?", "What is a supernova?",
]
FOURTH_ML = [
    "What model should I use to predict whether a guest cancels a hotel booking?",
    "How should I classify bank transactions as business or personal?",
    "Which model predicts if a pipe will leak?",
    "What should I use to predict whether a reader finishes an article?",
    "How do I predict if a taxi ride gets a tip?",
    "What classifier should I use to flag toxic comments?",
    "Which model is best to predict whether a flight is overbooked?",
    "How should I model whether a trial user converts?",
    "What method predicts if a loan application is fake?",
    "Which algorithm should I start with to predict hospital no-shows?",
    "How do I predict whether a customer answers a survey?",
    "What model should I use to tell if an image is blurry or sharp?",
    "How should I predict if a bus will arrive on time?",
    "What is a reasonable first model for a pass/fail label?",
    "Which model predicts whether a tenant pays rent late?",
    "How do I predict if a listener skips a song?",
    "What model should I use to detect whether a login is suspicious?",
    "How do I estimate whether a product gets returned?",
    "Which classifier would you pick to predict whether a movie wins an award?",
    "What should I use to predict whether a pump fails next month?",
]
VARIANTS = ["none", "bundle", "contrast", "midpoint"]


class DraftGated(s10.Gated):
    def __init__(self, q):
        super().__init__(q)
        self.pos = [[] for _ in range(R)]     # gist ids of drafts the rule went on to act on
        self.negd = [[] for _ in range(R)]    # ... and of drafts it left alone
        self.acted = []                       # per practice question, the rules that acted
        self.dkey, self.dtau = {}, {}

    def rehearse(self, question, gap):
        g0, m0 = len(self.gists), len(self.mom)
        reply = super().rehearse(question, gap)
        cuts = [(g, k) for _, g, k, _ in self.mom[m0:]]          # in order: draft g was cut by rule k
        for g in range(g0, len(self.gists)):
            active = {k for gc, k in cuts if gc < g}
            for k in range(R):
                if k in active:
                    continue
                (self.pos if any(kc == k for _, kc in cuts) else self.negd)[k].append(g)
        self.acted.append(sorted({k for _, k in cuts}))
        return reply

    def build_draft(self, kind, min_n=10, pct=99.5):
        G = np.array(self.gists)
        self.dkey, self.dtau, notes = {}, {}, []
        for k in range(R):
            if kind == "none" or len(self.pos[k]) < min_n or len(self.negd[k]) < min_n:
                continue
            P, N = G[self.pos[k]], G[self.negd[k]]
            key = unit(P.sum(0)) if kind == "bundle" else unit(P.mean(0) - N.mean(0))
            sp, sn = P @ key, N @ key
            tau = (sp.mean() + sn.mean()) / 2 if kind == "midpoint" else np.percentile(sn, pct)
            self.dkey[k], self.dtau[k] = key, float(tau)
            notes.append((k, len(P), len(N), round(float(tau), 2), round(float(np.mean(sp >= tau)), 2),
                          round(float(np.mean(sn >= tau)), 2)))
        return notes

    def reply(self, question):
        K, own = np.array(self.keys), np.array(self.owner)
        active, prefix, d, halts = [], [], [], []
        while True:
            d, _ = s8.draft(self.q, self.prompt(question, active), prefix, s8.MAX_NEW - len(prefix))
            g = self.gist(question, prefix + d)
            cut, best = None, 0.0
            for k, key in self.dkey.items():                     # the draft gates first
                if k not in active and float(key @ g) - self.dtau[k] >= best:
                    cut, best = (0, k), float(key @ g) - self.dtau[k]
            for i in range(len(d) if cut is None else 0):
                s = K @ self.key(self.situation(d[i:], prefix + d[:i]), g) - self.tau
                s[np.isin(own, active)] = -1
                j = int(s.argmax())
                if s[j] >= 0:
                    cut = (i, int(own[j]))
                    break
            if cut is None:
                break
            halts.append((len(prefix) + cut[0], cut[1]))
            prefix, d = prefix + d[:cut[0]], []
            active.append(cut[1])
        return self.q.tok.decode([t for t in prefix + d if t not in self.end]).strip(), sorted(active), halts


def main(gap, show, out):
    q = qwen.Qwen()
    t = time.time()
    s7.ASKS = s8.ASKS
    qs, _ = s8.practice(q)
    test = {x.lower() for x in s9.NEW_GENERAL + s9.NEW_ML + s10.THIRD_GENERAL + s10.THIRD_ML
            + FOURTH_GENERAL + FOURTH_ML}
    every = qs + s8.tempting(q, qs)[0]
    qs = [x for x in every if x.lower() not in test]
    rules = DraftGated(q)
    taught = [(question, rules.rehearse(question, gap)) for question in qs]
    print(f"{qwen.NAME}; {len(qs)} practice questions ({len(every) - len(qs)} equal to a test question, dropped), "
          f"rehearsed in {time.time() - t:.0f}s: {len(rules.mom)} moments; practice questions a rule acted on: "
          f"{[sum(k in a for a in rules.acted) for k in range(R)]} of {len(qs)}")
    print("  the ban acted on, e.g.: " + " | ".join(x for x, a in zip(qs, rules.acted) if BAN in a)[:900])
    print("  the ban left alone, e.g.: " + " | ".join(x for x, a in zip(qs, rules.acted) if BAN not in a)[:600] + "\n")
    rules.build(True, "rule_x")                                   # step10's pick, unchanged
    results = {"practice": qs, "rehearsed": taught, "acted": rules.acted, "points": rules.notes}

    print("dev (80 used questions): pick the draft gate")
    dev_g, dev_m = GENERAL + s9.NEW_GENERAL, ML + s9.NEW_ML
    best = None
    for kind in VARIANTS:
        gates = rules.build_draft(kind)
        rows, m = run(q, rules, "recall", dev_g, dev_m)
        score = m["signoff"] + m["ban"] - m["view"][BAN][0]
        print(line(kind, m) + f"  score {score:.2f}  draft gates (rule, pos, neg, tau, pos over, neg over): {gates}")
        results[f"dev_{kind}"] = {"gates": gates, "rows": rows, "score": score}
        if best is None or score > best[0]:
            best = (score, kind)
    kind = best[1]
    results["picked"] = {"kind": kind, "gates": rules.build_draft(kind)}
    print(f"picked: {kind}\n")

    print("test questions: fourth (never used)")
    for arm in ("alone", "prompt", "recall"):
        rows, m = run(q, rules, arm, FOURTH_GENERAL, FOURTH_ML)
        print(line(arm, m))
        if arm == "recall":
            for r in rows[:show] + rows[len(FOURTH_GENERAL):]:
                print(f"      {r['halts']} {r['q']} -> {r['reply']!r}")
        results[f"fourth_{arm}"] = {"rows": rows, **m}
        with open(os.path.join(HERE, out), "w") as fh:
            json.dump(results, fh, indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=8.0)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--out", default="step11_results.json")
    a = ap.parse_args()
    main(a.gap, a.show, a.out)
