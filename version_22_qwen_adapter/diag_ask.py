"""diag_ask.py — why does "Ask {p} what their {a} is named, and they will say"
drop to ~85%? For each answer token of each wording, the best key similarity
against tau and whether the best key's counts favour the right token.
Dev facts only (seed 1), so nothing here is picked on the test facts.

Run:  python diag_ask.py [--facts 600] [--sample 150] [--delta 0.05]
"""

import argparse
from collections import defaultdict

import numpy as np

import qwen
import step3_glyphh as s3
from step4_bigger import ASK, make_facts


def main(n, sample, delta, tau_pct=99.5):
    q = qwen.Qwen()
    from step4_bigger import calibrate
    tau = calibrate(q, pct=tau_pct, n_dev=n, delta=delta)
    facts = make_facts(n, seed=1)
    m = s3.GlyphhMemory(q.tok, tau, delta=delta)
    for f in facts:
        m.learn(q.encode(f["taught"] + f["answer"]))
    stats = defaultdict(lambda: defaultdict(list))   # key -> token index -> [(sim, right)]
    for f in facts[:sample]:
        for key in ASK:
            a = len(q.encode(f[key]))
            toks = q.encode(f[key] + f["answer"])
            ctx = m.contexts(toks)
            js, ss = m.nearest(ctx[a - 1:len(toks) - 1])
            for i, (j, s) in enumerate(zip(js, ss)):
                c = m.counts[j]
                want = toks[a + i]
                stats[key][i].append((s, c.get(want, 0) == max(c.values())))
    for key in ASK:
        print(f"{key:<8}", end="")
        for i in sorted(stats[key])[:5]:
            v = np.array(stats[key][i], dtype=float)
            print(f"  tok{i}: sim {np.median(v[:, 0]):.3f} >tau {np.mean(v[:, 0] >= tau):4.0%} right {np.mean(v[:, 1]):4.0%}", end="")
        print()
    # taught first-token misses: what the key was and what it predicted
    shown = 0
    for f in facts[:sample]:
        toks = q.encode(f["taught"] + f["answer"])
        a = len(q.encode(f["taught"]))
        j, s = m.nearest(m.contexts(toks)[a - 1:a])
        c = m.counts[j[0]]
        if c.get(toks[a], 0) != max(c.values()) and shown < 4:
            shown += 1
            print("miss:", repr(f["taught"] + f["answer"]), {q.tok.decode([k]): v for k, v in c.items()})
    # one example of what the window sees
    f = facts[0]
    for key in ("taught", "ask"):
        toks = q.encode(f[key] + f["answer"])
        text = q.tok.decode(toks[:len(q.encode(f[key])) + 1]).lower()
        words = [w for w in s3._WORD.findall(text) if w not in s3.STOP][-s3.WINDOW:]
        print(f"{key} window at 2nd name token: {words}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", type=int, default=600)
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--function-words", action="store_true")
    a = ap.parse_args()
    if a.function_words:
        s3.STOP |= s3.FUNCTION
    main(a.facts, a.sample, a.delta)
