"""diag_rehalt.py — step8d: why don't the ' logistic' gates halt the redraft again?

Rebuilds the step8c/d rule memory, then for every test reply of step8d that
says logistic: at each ' logistic' in it, the best gate and its similarity,
the similarity of the ' logistic' gates, and what their push would choose.
"""
import json
import os

import numpy as np

import qwen
import step7_rehearse as s7
import step8_lookback as s8
from step1_adapter import HERE
from step5_instructions import BRIEF, RULES, chat

q = qwen.Qwen()
s7.ASKS = s8.ASKS
qs, _ = s8.practice(q)
qs += s8.tempting(q, qs)[0]
rules = s8.Lookback(q)
for question in qs:
    rp = chat(q, [{"role": "system", "content": BRIEF + " " + RULES}, {"role": "user", "content": question}])
    pp = chat(q, [{"role": "system", "content": BRIEF}, {"role": "user", "content": question}])
    rules.rehearse(rp, pp, 8.0)
notes = rules.bundle()
rules.tau = rules.calibrate()
K = np.array(rules.keys)
gates = [j for j, x in enumerate(rules.xs) if "logistic" in q.tok.decode([x])]
print("tau", round(rules.tau, 3), "logistic gates:", [notes[j] for j in gates])

rows = json.load(open(os.path.join(HERE, "step8d_results.json")))["lookback"]
for r in rows:
    if not r.get("logistic"):
        continue
    pt = q.encode(chat(q, [{"role": "system", "content": BRIEF}, {"role": "user", "content": r["q"]}]))
    toks = q.encode(r["reply"])
    lp = rules.scored(pt + toks, len(pt) - 1, len(toks))
    for i, t in enumerate(toks):
        if "logistic" not in q.tok.decode([t]):
            continue
        s = K @ rules.situation(toks[i:], toks[:i])
        b = int(s.argmax())
        out = []
        for j in gates:
            lg = lp[i].copy()
            lg[rules.xs[j]] -= rules.downs[j]
            lg[rules.ys[j]] += rules.ups[j]
            out.append((round(float(s[j]), 2), q.tok.decode([int(lg.argmax())])))
        print(f"  ...{q.tok.decode(toks[max(0, i - 4):i])!r} | best gate {notes[b][:2]} {s[b]:.2f} | "
              f"logistic gates (sim, would choose): {out}")
