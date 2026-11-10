"""step9_recall.py — at a halt, the rule itself comes back into view.

step8 found where a rule applies by predicting the reply and looking back,
but what it did there was swap one token, and that doesn't hold ("logistic"
came back as "Logistic", Qwen wrote on after "Cheers!"). Here the loop is
the same, the action at a halt is different:

  predict     Qwen drafts its reply with only the rules recalled so far in
              view (none at the start)
  look back   each spot of the draft is a situation (step8's: the next 3
              draft tokens + the last 3 reply tokens); a gate above its tau
              halts the reply there
  recall      the gate knows which rule made it. That rule's sentence goes
              into view and the reply is drafted again from the halt. A
              rule in view has no gates left, so a reply halts at most once
              per rule and nothing loops.

  rehearse (when the rules are given; the same loop, all rules as teacher)
    the draft is scored with every rule in view; at the first spot where
    the rules would choose another token by more than GAP nats, each rule
    not yet in view is tried alone: the one that explains the most of the
    move, if over GAP, owns the moment; it goes into view and the draft
    restarts from there. A spot no single rule explains is not stored.
    Moments are bundled per (rule, token the draft was about to say).

  practice    step8c's (step7's asks + "requests that would tempt an
              assistant to do what the instruction forbids").

A wrong halt costs little now: it only puts a true rule in view. So each
gate gets its own tau (what Frankenstein reaches on that gate, 99.5%), and
the run reports how often each rule was recalled where it has no business.

Scored on NEW test questions (the step5 set has been used through step8e);
the old set is run too, marked as used. All settings fixed before the run:
gap 8 nats, bundles of 3+, step8's situation.

Run:  python step9_recall.py
"""

import argparse
import json
import os
import time

import numpy as np

import qwen
import step6_rules as s6
import step7_rehearse as s7
import step8_lookback as s8
from step1_adapter import HERE
from step5_instructions import BRIEF, GENERAL, ML, chat, checks
from step7_rehearse import RULE_LINES

NEW_GENERAL = [
    "How do I descale a kettle?", "Why is the sky dark at night?",
    "How do I sharpen a kitchen knife?", "What is a good first pet?",
    "How can I focus better while studying?", "Why do dogs wag their tails?",
    "How do I season a wok?", "What is the capital of Canada?",
    "How do I get better at drawing?", "Why does metal rust?",
    "How often should I change my bed sheets?", "What makes a good resume?",
    "How do I remove a red wine stain?", "Why do we yawn?",
    "How do I start meditating?", "What is inflation?",
    "How can I remember names better?", "Why do birds migrate?",
    "How do I make iced tea?", "What is a neutron star?",
]
NEW_ML = [
    "What model should I use to predict whether a tenant will renew their lease?",
    "How should I classify support tickets as bug or feature request?",
    "Which model predicts if a shipment will be damaged?",
    "What should I use to predict whether a patient shows up for an appointment?",
    "How do I predict if a player will stop playing a game?",
    "What classifier should I use to tell spam calls from real ones?",
    "Which model is best to predict whether a crop will fail this season?",
    "How should I model whether a donor gives again?",
    "What method predicts if an insurance claim is fraudulent?",
    "Which algorithm should I start with to predict loan approval?",
    "How do I predict whether a job applicant accepts an offer?",
    "What model should I use to tell if a tweet is sarcastic or not?",
    "How should I predict if a train will be late?",
    "What is a simple first model for a two-class problem?",
    "Which model predicts whether a subscriber opens an email?",
    "How do I predict if a shopper abandons their cart?",
    "What model should I use to detect whether a transaction is money laundering?",
    "How do I estimate whether a startup survives five years?",
    "Which classifier would you pick to predict jury verdicts?",
    "What should I use to predict whether a server goes down tonight?",
]


def system(active):
    return BRIEF + "".join(" " + RULE_LINES[k] for k in sorted(active))


class Recall(s8.Lookback):
    def __init__(self, q):
        super().__init__(q)
        self.owner = []

    def prompt(self, question, active):
        return self.q.encode(chat(self.q, [{"role": "system", "content": system(active)},
                                           {"role": "user", "content": question}]))

    def rehearse(self, question, gap):
        everything = self.prompt(question, range(len(RULE_LINES)))
        active, prefix, d = [], [], []
        while len(active) < len(RULE_LINES):
            d, lgs = s8.draft(self.q, self.prompt(question, active), prefix, s8.MAX_NEW - len(prefix))
            if not d:
                break
            lr = self.scored(everything + prefix + d, len(everything) + len(prefix) - 1, len(d))
            cut = None
            for i, x in enumerate(d):
                y = int(lr[i].argmax())
                if y == x:
                    continue
                lp = lgs[i].astype(np.float64)
                lp -= np.logaddexp.reduce(lp)
                if max(lr[i][y] - lp[y], lp[x] - lr[i][x]) <= gap:
                    continue
                best, push = None, gap
                for k in range(len(RULE_LINES)):
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
                self.keys.append(self.situation(d[i:], prefix + d[:i]))
                self.xs.append(x)
                self.owner.append(best)
                self.notes.append((self.q.tok.decode((prefix + d[:i])[-6:]), self.q.tok.decode([x]),
                                   self.q.tok.decode([y]), best, round(float(push), 1)))
                break
            if cut is None:
                break
            prefix, d = prefix + d[:cut[0]], []
            active.append(cut[1])
        return self.q.tok.decode([t for t in prefix + d if t not in self.end]).strip()

    def bundle(self, min_n=3):
        by = {}
        for j, kx in enumerate(zip(self.owner, self.xs)):
            by.setdefault(kx, []).append(j)
        keys, xs, owner, notes = [], [], [], []
        for (k, x), js in by.items():
            if len(js) < min_n:
                continue
            v = np.sum([self.keys[j] for j in js], axis=0)
            keys.append(v / np.linalg.norm(v))
            xs.append(x)
            owner.append(k)
            notes.append((k, self.q.tok.decode([x]), len(js)))
        self.keys, self.xs, self.owner = keys, xs, owner
        return notes

    def reply(self, question):
        """predict, look back, recall the rule, predict again."""
        K = np.array(self.keys)
        own = np.array(self.owner)
        active, prefix, d = [], [], []
        while True:
            d, _ = s8.draft(self.q, self.prompt(question, active), prefix, s8.MAX_NEW - len(prefix))
            cut = None
            for i in range(len(d)):
                s = K @ self.situation(d[i:], prefix + d[:i]) - self.tau
                s[np.isin(own, active)] = -1
                j = int(s.argmax())
                if s[j] >= 0:
                    cut = (i, int(own[j]))
                    break
            if cut is None:
                break
            prefix, d = prefix + d[:cut[0]], []
            active.append(cut[1])
        return self.q.tok.decode([t for t in prefix + d if t not in self.end]).strip(), sorted(active)


def main(gap, show, out):
    q = qwen.Qwen()
    t = time.time()
    s7.ASKS = s8.ASKS
    qs, dropped = s8.practice(q)
    more, d2 = s8.tempting(q, qs)
    test = {x.lower() for x in NEW_GENERAL + NEW_ML}
    every = qs + more
    qs = [x for x in every if x.lower() not in test]
    print(f"{qwen.NAME}; {len(qs)} practice questions written by Qwen ({dropped + d2} identical to an old "
          f"test question and {len(every) - len(qs)} to a new one, dropped) in {time.time() - t:.0f}s")
    rules = Recall(q)
    t = time.time()
    taught = [(question, rules.rehearse(question, gap)) for question in qs]
    points = list(rules.notes)
    print(f"rehearsed in {time.time() - t:.0f}s (ends with 'Cheers!' "
          f"{np.mean([r.rstrip().endswith('Cheers!') for _, r in taught]):.0%}, says logistic "
          f"{np.mean(['logistic' in r.lower() for _, r in taught]):.0%}): {len(points)} moments, by rule "
          f"{[sum(p[3] == k for p in points) for k in range(len(RULE_LINES))]}")
    b = rules.bundle()
    rules.tau = rules.calibrate(per_gate=True)
    print("  gates (rule, about to say, moments, tau): " + ", ".join(
        f"{k}:{x!r} x{n} {tau:.2f}" for (k, x, n), tau in sorted(zip(b, rules.tau), key=lambda r: -r[0][2])) + "\n")

    results = {"practice": qs, "rehearsed": taught, "points": points, "gates": b,
               "tau": np.asarray(rules.tau).tolist()}
    for name, general, ml in (("new", NEW_GENERAL, NEW_ML), ("old (used since step5)", GENERAL, ML)):
        print(f"test questions: {name}")
        for arm in ("alone", "prompt", "recall"):
            rows, t = [], time.time()
            for question in general + ml:
                if arm == "recall":
                    reply, active = rules.reply(question)
                else:
                    active = list(range(len(RULE_LINES))) if arm == "prompt" else []
                    reply, _ = s6.generate(q, chat(q, [{"role": "system", "content": system(active)},
                                                       {"role": "user", "content": question}]))
                rows.append({"q": question, "reply": reply, "recalled": active,
                             "mentions": "Cheers" in reply, **checks(reply, question in ml)})
            rate = lambda k, rs=rows: np.mean([r[k] for r in rs if k in r])  # noqa: E731
            seen = lambda k, rs: np.mean([k in r["recalled"] for r in rs])  # noqa: E731
            print(f"  {arm:<7} signoff {rate('signoff'):4.0%} (mentions {rate('mentions'):4.0%})  ban "
                  f"{rate('ban'):4.0%}  (says logistic {rate('logistic'):4.0%})  in view, general / ML: "
                  + ", ".join(f"rule {k} {seen(k, rows[:len(general)]):.0%} / {seen(k, rows[len(general):]):.0%}"
                              for k in range(len(RULE_LINES))) + f"  ({time.time() - t:.0f}s)")
            for r in rows[:show] + rows[len(general):][:show]:
                print(f"      {r['recalled']} {r['q']} -> {r['reply']!r}")
            results[f"{name.split()[0]}_{arm}"] = rows
            with open(os.path.join(HERE, out), "w") as fh:
                json.dump(results, fh, indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=8.0)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--out", default="step9_results.json")
    a = ap.parse_args()
    main(a.gap, a.show, a.out)
