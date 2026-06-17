"""
Head-to-head: Adam vs BeliefOpt (the 'trend-belief' optimizer = AdaBelief).

Same model, seed/init, batches, step budget; only the optimizer differs.
BeliefOpt scales each step by the gradient's deviation from its own trend
(s = EMA((g-m)^2)) instead of by the gradient magnitude (Adam's v = EMA(g^2)):
push harder when the nudge is consistent with its trend, slow down when it's
volatile relative to that trend.

Metric on this (saturating toy) corpus: steps-to-90% val acc (= convergence
speed) plus final val loss/acc, averaged over seeds with error bars.

Run:  python compare_opt.py
"""

import numpy as np
import step3_gpt as G

STEPS = 800
SEEDS = (0, 1, 2)
EVAL_EVERY = 25


def steps_to(hist, target=0.90):
    for step, _, _, _, va in hist:
        if va >= target:
            return step
    return None


def run(make_opt, seed):
    p, hist = G.train(steps=STEPS, B=32, seed=seed, make_opt=make_opt,
                      eval_every=EVAL_EVERY, verbose=False)
    eval_rng = np.random.default_rng(123)
    va = [G.evaluate(p, "val", eval_rng) for _ in range(8)]
    return dict(hist=hist,
                va_loss=np.mean([l for l, _ in va]),
                va_acc=np.mean([a for _, a in va]),
                hit90=steps_to(hist, 0.90))


def main():
    print(f"vocab={G.vocab_size} C={G.C} T={G.BLOCK}  |  {STEPS} steps, seeds {SEEDS}\n")
    opts = {
        "Adam":      lambda p: G.Adam(p, lr=3e-3),
        "BeliefOpt": lambda p: G.BeliefOpt(p, lr=3e-3),
    }
    res = {n: [run(mk, s) for s in SEEDS] for n, mk in opts.items()}

    # seed-0 curves
    print(f"seed {SEEDS[0]} val loss / val acc:")
    print(f"{'step':>5} | {'Adam':>16} | {'BeliefOpt':>16}")
    print("-" * 46)
    for a, b in zip(res["Adam"][0]["hist"], res["BeliefOpt"][0]["hist"]):
        if a[0] == 1 or a[0] % 100 == 0:
            print(f"{a[0]:>5} | {a[3]:7.3f} {a[4]:6.1%} | {b[3]:7.3f} {b[4]:6.1%}")

    print(f"\naveraged over seeds {SEEDS}:")
    print(f"{'':12} {'val loss (mean±std)':>22} {'val acc':>9} {'->90% @ (mean)':>15}")
    for name in opts:
        rs = res[name]
        vl = [r["va_loss"] for r in rs]
        va = np.mean([r["va_acc"] for r in rs])
        h = [r["hit90"] for r in rs if r["hit90"]]
        hit = f"{np.mean(h):.0f}" if h else "never"
        print(f"{name:12} {np.mean(vl):10.3f} ± {np.std(vl):5.3f}      "
              f"{va:8.1%} {hit:>15}")


if __name__ == "__main__":
    main()
