"""diag_corrections.py — why do corrections reach only half of "is called" and
"answers to"? For corrected facts, show which stored glyphh each wording's
answer tokens land on, how similar it is, and what that key's counts say.

Run:  python diag_corrections.py [--facts 120] [--show 4]
"""

import argparse
from collections import Counter

import numpy as np

import qwen
import step3_glyphh as s3
from step4_bigger import ASK, calibrate, make_facts


def main(n, show):
    q = qwen.Qwen()
    tau = calibrate(q)
    m = s3.GlyphhMemory(q.tok, tau)
    src = []                                        # row -> (sentence, position)

    def teach(text):
        toks = q.encode(text)
        before = len(m.counts)
        m.learn(toks)
        ctx = m.contexts(toks)
        rows = {m._index[ctx[t].tobytes()]: t for t in range(len(toks) - 1) if ctx[t].any()}
        for r in range(before, len(m.counts)):
            src.append((text, rows.get(r, -1)))

    facts = make_facts(n)
    for f in facts:
        teach(f["taught"] + f["answer"])
    fixed = facts[:n // 3]
    for f in fixed:
        teach(f["taught"] + f["new"])

    dec = lambda ids: q.tok.decode(ids)  # noqa: E731
    tally = {k: Counter() for k in ASK}
    for i, f in enumerate(fixed):
        for key in ASK:
            prompt = f[key]
            a = len(q.encode(prompt))
            toks = q.encode(prompt + f["new"])
            ctx = m.contexts(toks)
            js, ss = m.nearest(ctx[a - 1:len(toks) - 1])
            for step, (j, s) in enumerate(zip(js, ss)):
                t = a - 1 + step
                want = toks[t + 1]
                text, pos = src[j]
                c = m.counts[j]
                same_fact = text.startswith(f["taught"])
                verdict = ("below tau" if s < tau else
                           "right" if c.get(want, 0) == max(c.values()) else
                           "other fact" if not same_fact else "same fact, wrong count")
                if step == 0:
                    tally[key][verdict] += 1
                if i < show:
                    stem = q.tok.decode(q.encode(text)[:pos + 1]) if pos >= 0 else "?"
                    print(f"[{key:<7}] predicting {dec([want])!r:<8} hit {s:.3f} -> "
                          f"key after {stem[-48:]!r}  counts {{{', '.join(f'{dec([k])!r}:{v}' for k, v in c.items())}}}  "
                          f"=> {verdict}")
        if i < show:
            print()
    print("first answer token, over", len(fixed), "corrected facts:")
    for key, c in tally.items():
        print(f"  {key:<8} " + ", ".join(f"{v} {k}" for k, v in c.most_common()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", type=int, default=120)
    ap.add_argument("--show", type=int, default=3)
    a = ap.parse_args()
    main(a.facts, a.show)
