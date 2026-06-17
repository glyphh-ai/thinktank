"""
Does the sampling gap GROW with model capacity?

Same uniform-vs-prioritized comparison, but on a bigger model + longer context
(set_config: C 64->128, context 16->32, ~59k->219k params). More capacity and
more context = more headroom, so if prioritized sampling is real its advantage
should widen here vs the base model.

Run:  python compare_scaled.py
"""

import numpy as np
import step3_gpt as G

# --- scale up BEFORE building anything ---
CFG = G.set_config(width=128, ctx=32)

STEPS = 1500
SEEDS = (0, 1)
EVAL_EVERY = 100


def run(make_sampler, seed):
    sampler = make_sampler(seed)
    _, hist = G.train(steps=STEPS, B=32, seed=seed, sampler=sampler,
                      eval_every=EVAL_EVERY, verbose=False)
    return hist


def final_loss(h): return h[-1][3]
def final_acc(h):  return h[-1][4]
def best_loss(h):  return min(r[3] for r in h)


def main():
    n_tot = len(G.train_data) + len(G.val_data)
    print(f"corpus={n_tot:,} chars  vocab={G.vocab_size}  "
          f"C={CFG['C']} T={CFG['CTX']}  ~219k params  |  {STEPS} steps, seeds {SEEDS}\n")
    makers = {
        "uniform":     lambda s: G.UniformSampler("train", B=32, seed=s),
        "prioritized": lambda s: G.PrioritizedSampler("train", B=32, seed=s,
                                                      beta=0.9, eps=0.1),
    }
    res = {n: [run(mk, s) for s in SEEDS] for n, mk in makers.items()}

    # seed-0 curve
    print(f"seed {SEEDS[0]} val loss / val acc:")
    print(f"{'step':>5} | {'uniform':>16} | {'prioritized':>16}")
    print("-" * 46)
    for u, p in zip(res["uniform"][0], res["prioritized"][0]):
        if u[0] == 1 or u[0] % 300 == 0:
            print(f"{u[0]:>5} | {u[3]:7.3f} {u[4]:6.1%} | {p[3]:7.3f} {p[4]:6.1%}")

    print(f"\naveraged over seeds {SEEDS}:")
    print(f"{'':12} {'final val loss':>15} {'final val acc':>14} {'best val loss':>14}")
    for name in makers:
        hs = res[name]
        fl = np.mean([final_loss(h) for h in hs])
        fa = np.mean([final_acc(h) for h in hs])
        bl = np.mean([best_loss(h) for h in hs])
        print(f"{name:12} {fl:15.3f} {fa:14.1%} {bl:14.3f}")


if __name__ == "__main__":
    main()
