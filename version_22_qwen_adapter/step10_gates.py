"""step10_gates.py — gates that fire only where their rule acts.

step9 recalled the right thing at a halt (the rule's sentence) and the rules
carried, but every rule was recalled on every reply: the gates fired on
everything. Two reasons, both fixed here from the rehearsal itself:

  the key    a situation was local to the reply (next 3 draft tokens + last
             3 written). "To predict if an employee will quit, ..." and "To
             descale a kettle, ..." start the same. The prediction is the
             whole draft, so the key can carry it: + a GIST glyphh, the
             content words of the question and the predicted reply.
  the tau    was set on Frankenstein, which is nothing like a chat reply.
             Rehearsal already holds the right negatives: every spot of
             every practice draft where a rule was not in view and the
             rules changed nothing. A gate's tau is what THOSE spots reach
             on it (99.5%). No outside text.

Rehearsal, recall at a halt and practice are step9's. Four gate variants
(gist on / off  x  one gate per rule / one per (rule, token about to be
said)) are compared on a DEV set: the old 40 questions, used since step5.
Picked before any test run, by
    sign-off + ban - (ban rule in view on general questions).
The pick is then run once on 40 questions never used (THIRD_*), and on
step9's 40 (used once).

Run:  python step10_gates.py
"""

import argparse
import json
import os
import time

import numpy as np

import qwen
import step3_glyphh as s3
import step6_rules as s6
import step7_rehearse as s7
import step8_lookback as s8
import step9_recall as s9
from step1_adapter import HERE
from step5_instructions import GENERAL, ML, chat, checks
from step7_rehearse import RULE_LINES

R = len(RULE_LINES)
BAN = 2
THIRD_GENERAL = [
    "How do I unclog a sink?", "Why is the sea blue?", "How do I iron a shirt?",
    "What is a good first musical instrument?", "How can I wake up earlier?",
    "Why do onions make you cry?", "How do I clean a keyboard?", "What is the capital of Japan?",
    "How do I get better at public speaking?", "Why does ice float?",
    "How often should I replace my toothbrush?", "What makes a good cover letter?",
    "How do I get gum out of hair?", "Why do we dream?", "How do I start journaling?",
    "What is a mortgage?", "How can I read faster?", "Why do stars twinkle?",
    "How do I make pancakes?", "What is a comet?",
]
THIRD_ML = [
    "What model should I use to predict whether a customer returns a product?",
    "How should I classify news articles as sports or politics?",
    "Which model predicts if a bridge needs repair?",
    "What should I use to predict whether a student drops out?",
    "How do I predict if a driver will file a claim?",
    "What classifier should I use to sort resumes into interview or reject?",
    "Which model is best to predict whether a store runs out of stock?",
    "How should I model whether a member cancels their gym membership?",
    "What method predicts if an email is phishing?",
    "Which algorithm should I start with to predict credit card default?",
    "How do I predict whether a voter supports a ballot measure?",
    "What model should I use to tell if a photo contains a cat or not?",
    "How should I predict if a delivery will be on time?",
    "What is a sensible starting model for yes/no predictions?",
    "Which model predicts whether a patient responds to a treatment?",
    "How do I predict if a user upgrades to the paid plan?",
    "What model should I use to detect whether a sensor reading is faulty?",
    "How do I estimate whether a restaurant passes inspection?",
    "Which classifier would you pick to predict whether a song becomes a hit?",
    "What should I use to predict whether a battery fails within a year?",
]
VARIANTS = [(True, "rule"), (True, "rule_x"), (False, "rule"), (False, "rule_x")]


def unit(v):
    return v / (np.linalg.norm(v) or 1.0)


class Gated(s9.Recall):
    def __init__(self, q):
        super().__init__(q)
        self.gists = []                 # one per practice draft
        self.mom = []                   # (local, gist id, rule, token about to be said)
        self.neg = []                   # (local, gist id, rules not in view) where nothing changed
        self.use_gist = True

    def gist(self, question, toks):
        words = s3._WORD.findall((question + " " + self.q.tok.decode(toks)).lower())
        v = sum((self.g._wglyph(w) for w in words if w not in s3.STOP), np.zeros(s3.DIM, np.float32))
        return unit(v)

    def key(self, local, gist):
        return unit(local + gist) if self.use_gist else local

    def rehearse(self, question, gap):
        everything = self.prompt(question, range(R))
        active, prefix, d = [], [], []
        while True:
            d, lgs = s8.draft(self.q, self.prompt(question, active), prefix, s8.MAX_NEW - len(prefix))
            if not d:
                break
            lr = self.scored(everything + prefix + d, len(everything) + len(prefix) - 1, len(d))
            self.gists.append(self.gist(question, prefix + d))
            g, cut, moved = len(self.gists) - 1, None, []
            for i, x in enumerate(d):
                if len(active) == R:
                    break
                y = int(lr[i].argmax())
                if y == x:
                    continue
                lp = lgs[i].astype(np.float64)
                lp -= np.logaddexp.reduce(lp)
                if max(lr[i][y] - lp[y], lp[x] - lr[i][x]) <= gap:
                    continue
                moved.append(i)
                best, push = None, gap
                for k in range(R):
                    if k in active:
                        continue
                    pk = self.prompt(question, active + [k]) + prefix + d[:i]
                    lk = self.scored(pk, len(pk) - 1, 1)[0]
                    explains = (lk[y] - lp[y]) + (lp[x] - lk[x])
                    if explains > push:
                        best, push = k, explains
                if best is None:
                    continue
                cut = (i, best)
                self.mom.append((self.situation(d[i:], prefix + d[:i]), g, best, x))
                self.notes.append((self.q.tok.decode((prefix + d[:i])[-6:]), self.q.tok.decode([x]),
                                   self.q.tok.decode([y]), best, round(float(push), 1)))
                break
            # negatives: spots the rules left alone (not a spot they moved, nor the
            # AHEAD-1 before it, which already see it coming), every 2nd one
            quiet = [k not in active for k in range(R)]
            near = {i - a for i in moved for a in range(s8.AHEAD)}
            for i in range(0, len(d) if cut is None else cut[0], 2):
                if i not in near and any(quiet):
                    self.neg.append((self.situation(d[i:], prefix + d[:i]), g, quiet))
            if cut is None:
                break
            prefix, d = prefix + d[:cut[0]], []
            active.append(cut[1])
        return self.q.tok.decode([t for t in prefix + d if t not in self.end]).strip()

    def build(self, use_gist, per, min_n=3, pct=99.5):
        """the gates of one variant, each with a tau from the rehearsal negatives."""
        self.use_gist = use_gist
        by = {}
        for j, (_, _, k, x) in enumerate(self.mom):
            by.setdefault((k, x) if per == "rule_x" else (k, None), []).append(j)
        G = np.array(self.gists)
        N = np.array([self.key(v, G[g]) for v, g, _ in self.neg])
        quiet = np.array([m for _, _, m in self.neg])
        self.keys, self.owner, self.tau, notes = [], [], [], []
        for (k, x), js in by.items():
            if len(js) < min_n:
                continue
            P = np.array([self.key(self.mom[j][0], G[self.mom[j][1]]) for j in js])
            key = unit(P.sum(0))
            tau = float(np.percentile(N[quiet[:, k]] @ key, pct))
            self.keys.append(key)
            self.owner.append(k)
            self.tau.append(tau)
            notes.append((k, "*" if x is None else self.q.tok.decode([x]), len(js), round(tau, 2),
                          round(float(np.mean(P @ key >= tau)), 2)))
        self.tau = np.array(self.tau)
        return notes

    def reply(self, question):
        K, own = np.array(self.keys), np.array(self.owner)
        active, prefix, d, halts = [], [], [], []
        while True:
            d, _ = s8.draft(self.q, self.prompt(question, active), prefix, s8.MAX_NEW - len(prefix))
            g = self.gist(question, prefix + d)
            cut = None
            for i in range(len(d)):
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


def run(q, rules, arm, general, ml):
    rows = []
    for question in general + ml:
        if arm == "recall":
            reply, active, halts = rules.reply(question)
        else:
            active, halts = (list(range(R)) if arm == "prompt" else []), []
            reply, _ = s6.generate(q, chat(q, [{"role": "system", "content": s9.system(active)},
                                               {"role": "user", "content": question}]))
        rows.append({"q": question, "reply": reply, "recalled": active, "halts": halts,
                     "mentions": "Cheers" in reply, **checks(reply, question in ml)})
    rate = lambda k: float(np.mean([r[k] for r in rows if k in r]))  # noqa: E731
    seen = lambda k, rs: float(np.mean([k in r["recalled"] for r in rs]))  # noqa: E731
    view = [(seen(k, rows[:len(general)]), seen(k, rows[len(general):])) for k in range(R)]
    return rows, {"signoff": rate("signoff"), "ban": rate("ban"), "logistic": rate("logistic"), "view": view}


def line(arm, m):
    return (f"  {arm:<22} signoff {m['signoff']:4.0%}  ban {m['ban']:4.0%}  (says logistic {m['logistic']:4.0%})"
            "  in view, general / ML: " + ", ".join(f"rule {k} {a:.0%} / {b:.0%}" for k, (a, b) in enumerate(m["view"])))


def main(gap, show, out):
    q = qwen.Qwen()
    t = time.time()
    s7.ASKS = s8.ASKS
    qs, _ = s8.practice(q)
    test = {x.lower() for x in s9.NEW_GENERAL + s9.NEW_ML + THIRD_GENERAL + THIRD_ML}
    every = qs + s8.tempting(q, qs)[0]
    qs = [x for x in every if x.lower() not in test]
    rules = Gated(q)
    taught = [(question, rules.rehearse(question, gap)) for question in qs]
    print(f"{qwen.NAME}; {len(qs)} practice questions ({len(every) - len(qs)} equal to a test question, dropped), "
          f"rehearsed in {time.time() - t:.0f}s: {len(rules.mom)} moments, by rule "
          f"{[sum(m[2] == k for m in rules.mom) for k in range(R)]}; {len(rules.neg)} quiet spots\n")
    results = {"practice": qs, "rehearsed": taught, "points": rules.notes}

    print("dev (the old 40 questions): pick the gates")
    best = None
    for use_gist, per in VARIANTS:
        gates = rules.build(use_gist, per)
        rows, m = run(q, rules, "recall", GENERAL, ML)
        score = m["signoff"] + m["ban"] - m["view"][BAN][0]
        name = f"{'gist' if use_gist else 'local'}, per {per}"
        print(line(name, m) + f"  score {score:.2f}  ({len(gates)} gates)")
        results[f"dev_{name}"] = {"gates": gates, "rows": rows, "score": score}
        if best is None or score > best[0]:
            best = (score, use_gist, per, name)
    _, use_gist, per, name = best
    gates = rules.build(use_gist, per)
    print(f"picked: {name}\n  gates (rule, about to say, moments, tau, own moments over tau): "
          + ", ".join(f"{k}:{x!r} x{n} {tau} {own:.0%}" for k, x, n, tau, own in sorted(gates, key=lambda r: -r[2])) + "\n")
    results["picked"] = {"name": name, "gates": gates}

    for label, general, ml in (("third (never used)", THIRD_GENERAL, THIRD_ML),
                               ("step9's (used once)", s9.NEW_GENERAL, s9.NEW_ML)):
        print(f"test questions: {label}")
        for arm in ("alone", "prompt", "recall"):
            rows, m = run(q, rules, arm, general, ml)
            print(line(arm, m))
            if arm == "recall":
                for r in rows[:show] + rows[len(general):][:show]:
                    print(f"      {r['halts']} {r['q']} -> {r['reply']!r}")
            results[f"{label.split()[0]}_{arm}"] = {"rows": rows, **m}
            with open(os.path.join(HERE, out), "w") as fh:
                json.dump(results, fh, indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=8.0)
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--out", default="step10_results.json")
    a = ap.parse_args()
    main(a.gap, a.show, a.out)
