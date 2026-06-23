"""
head_to_head.py — baseline Adam vs Markov sign-reversion Adam.

Same model, same data, same seeds. The only difference is whether the
per-weight Markov stay-probability gain (Adam(reversion=True)) is on. We run a
few seeds each so the comparison isn't a single-draw fluke, and report final
train/val loss + accuracy averaged over seeds.

Run:  python head_to_head.py
"""

import numpy as np
import step3_gpt as G

STEPS = 700
SEEDS = [0, 1, 2]


def run(reversion, alpha=1.0):
    rows = []
    for seed in SEEDS:
        p, _ = G.train(steps=STEPS, seed=seed, verbose=False,
                       reversion=reversion, alpha=alpha)
        eval_rng = np.random.default_rng(999)
        tr_loss, tr_acc = G.evaluate(p, "train", eval_rng)
        va_loss, va_acc = G.evaluate(p, "val", eval_rng)
        rows.append((tr_loss, tr_acc, va_loss, va_acc))
    return np.array(rows)


def summarize(name, rows):
    m = rows.mean(0)
    print(f"{name:24s} | train loss {m[0]:6.3f} acc {m[1]:6.1%}"
          f" | val loss {m[2]:6.3f} acc {m[3]:6.1%}")
    return m


if __name__ == "__main__":
    print(f"head-to-head over seeds {SEEDS}, {STEPS} steps each\n")
    base = run(reversion=False)
    rev = run(reversion=True, alpha=1.0)
    print()
    mb = summarize("baseline Adam", base)
    mr = summarize("reversion Adam (a=1.0)", rev)
    print()
    dv_loss = mr[2] - mb[2]
    dv_acc = mr[3] - mb[3]
    verdict = "reversion WINS" if (dv_acc > 0 and dv_loss < 0) else (
              "baseline WINS" if (dv_acc < 0 and dv_loss > 0) else "MIXED")
    print(f"val delta (reversion - baseline): loss {dv_loss:+.3f}  acc {dv_acc:+.1%}"
          f"   -> {verdict}")
