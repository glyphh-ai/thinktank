"""step7_rehearse.py — the rule channel, rehearsed when the rule is taught.

step6 learned rule nudges from six session-1 replies: too few situations,
so new replies rarely landed on one. Here, when the rules are given, the
system rehearses them on its own:

  practice questions  Qwen writes them itself from each rule ("write
                      questions where this instruction would matter") plus
                      everyday questions; none are the test questions (any
                      exact overlap is dropped and counted)
  rehearse            Qwen answers each with the rules in view; step6's
                      decision points are harvested from every answer
  apply               session 2 as in step5/6: new questions, no rules in
                      the prompt, the nudges on

Settings are step6's, fixed before this run (gap 8 nats, top-5 + last 3
tokens, tau from Frankenstein); nothing is tuned on the test questions. One
change, decided before running: a nudged token fires at most once per reply
(not once per stored point), since with many stored points the "about to
end -> Cheers!" loop could otherwise restart from another one.

Run:  python step7_rehearse.py
"""

import argparse
import json
import os
import re
import time

import numpy as np

import qwen
import step6_rules as s6
from step1_adapter import HERE
from step5_instructions import BRIEF, GENERAL, ML, RULES, chat, checks
from step5_instructions import generate as plain_generate

RULE_LINES = ["Call me Chris.", "End every answer with 'Cheers!'.",
              "Never recommend logistic regression; recommend a decision tree instead."]
ASKS = ["Write 40 different short questions a user might ask an AI assistant about "
        "everyday topics. One per line, no numbering."] + [
        f"Here is an instruction for an AI assistant: \"{r}\" Write 30 different short "
        f"questions a user might ask where this instruction would matter. One per line, "
        f"no numbering." for r in RULE_LINES]


class RehearsedRules(s6.RuleMemory):
    def nudge(self, logits, recent, p, spent):
        used = {self.tokens[j] for j in spent}
        keep = [j for j in range(len(self.keys)) if self.tokens[j] not in used]
        if not keep:
            return p, None
        K = np.array([self.keys[j] for j in keep])
        s = K @ self.situation(logits, recent)
        i = int(s.argmax())
        if s[i] < self.tau:
            return p, None
        j = keep[i]
        lg = np.log(np.maximum(p, 1e-30))
        lg[self.tokens[j]] += self.gaps[j]
        q = np.exp(lg - lg.max())
        return q / q.sum(), j


def bundle(rules, min_n=3):
    """step7b: bundle every decision point that chose the same token into one
    glyphh (their sum): what the situations share (about to end, a sentence
    just closed) adds up, the reply words that differ each time wash out. A
    token chosen fewer than min_n times is dropped as noise."""
    by = {}
    for j, t in enumerate(rules.tokens):
        by.setdefault(t, []).append(j)
    keys, toks, gaps, notes = [], [], [], []
    for t, js in by.items():
        if len(js) < min_n:
            continue
        v = np.sum([rules.keys[j] for j in js], axis=0)
        keys.append(v / np.linalg.norm(v))
        toks.append(t)
        gaps.append(float(np.median([rules.gaps[j] for j in js])))
        notes.append((rules.q.tok.decode([t]), len(js), round(gaps[-1], 1)))
    rules.keys, rules.tokens, rules.gaps = keys, toks, gaps
    return notes


def practice(q):
    test = {t.lower().strip() for t in GENERAL + ML}
    qs, dropped = [], 0
    for ask in ASKS:
        text, _ = plain_generate(q, chat(q, [{"role": "user", "content": ask}]), max_new=900)
        for line in text.splitlines():
            line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip()
            if not line.endswith("?") or len(line) < 10:
                continue
            if line.lower() in test:
                dropped += 1
                continue
            if line not in qs:
                qs.append(line)
    return qs, dropped


def main(gap, show, out, do_bundle=False):
    q = qwen.Qwen()
    t = time.time()
    qs, dropped = practice(q)
    print(f"{qwen.NAME}; {len(qs)} practice questions written by Qwen ({dropped} identical to a test "
          f"question, dropped) in {time.time() - t:.0f}s, e.g.:")
    for x in qs[::max(1, len(qs) // 8)][:8]:
        print("   ", x)
    rules = RehearsedRules(q)
    t = time.time()
    follow = []
    for question in qs:
        rp = chat(q, [{"role": "system", "content": BRIEF + " " + RULES}, {"role": "user", "content": question}])
        pp = chat(q, [{"role": "system", "content": BRIEF}, {"role": "user", "content": question}])
        reply, _ = plain_generate(q, rp)
        follow.append(reply.rstrip().endswith("Cheers!"))
        rules.learn(rp, pp, reply, gap)
    kinds = {}
    for _, _, chose, _ in rules.notes:
        kinds[chose] = kinds.get(chose, 0) + 1
    print(f"rehearsed in {time.time() - t:.0f}s (signed off in view {np.mean(follow):.0%}): "
          f"{len(rules.keys)} decision points; most common rule tokens: "
          + ", ".join(f"{k!r} {v}" for k, v in sorted(kinds.items(), key=lambda x: -x[1])[:10]))
    if do_bundle:
        b = bundle(rules)
        print("  bundled: " + ", ".join(f"{t!r} x{n} (+{g})" for t, n, g in b))
    rules.tau = s6.calibrate(q, rules)
    print(f"  tau {rules.tau:.3f}\n")

    results = {"practice": qs, "points": rules.notes, "tau": rules.tau}
    for arm in ("alone", "prompt", "rules"):
        system = BRIEF + (" " + RULES if arm == "prompt" else "")
        rows, t = [], time.time()
        for question in GENERAL + ML:
            prompt = chat(q, [{"role": "system", "content": system}, {"role": "user", "content": question}])
            reply, fired = s6.generate(q, prompt, rules if arm == "rules" else None)
            rows.append({"q": question, "reply": reply, "fired": fired, **checks(reply, question in ML)})
        rate = lambda k, rs=rows: np.mean([r[k] for r in rs if k in r])  # noqa: E731
        print(f"{arm:<7} signoff {rate('signoff'):4.0%}  ban {rate('ban'):4.0%}  (says logistic "
              f"{rate('logistic'):4.0%})  nudges per reply {rate('fired'):.1f}  ({time.time() - t:.0f}s)")
        for r in rows[:show] + [r for r in rows if r["q"] in ML][:show]:
            print(f"    {r['q']} -> {r['reply']!r}")
        results[arm] = rows
        with open(os.path.join(HERE, out), "w") as fh:
            json.dump(results, fh, indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=8.0)
    ap.add_argument("--show", type=int, default=4)
    ap.add_argument("--out", default="step7_results.json")
    ap.add_argument("--bundle", action="store_true", help="step7b: one bundled glyphh per rule token")
    a = ap.parse_args()
    main(a.gap, a.show, a.out, a.bundle)
