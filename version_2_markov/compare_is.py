"""
Does importance-sampling (IS) correction rescue prioritized sampling?

Three-way, same model/seed/optimizer/budget, only the sampler differs:
  - uniform                      : baseline
  - prioritized (no IS)          : our original, biased toward hard windows
  - prioritized (IS-corrected)   : gradient reweighted by (1/M)/p_i so the
                                   expected gradient stays the uniform objective

If the distribution-bias hypothesis is right, IS correction should match (or
beat) uniform where the biased version fell short.

Run:  python compare_is.py
"""

import numpy as np
import step3_gpt as G

STEPS = 2000
SEEDS = (0, 1, 2)
EVAL_EVERY = 100


def run(make_sampler, seed):
    _, hist = G.train(steps=STEPS, B=32, seed=seed, sampler=make_sampler(seed),
                      eval_every=EVAL_EVERY, verbose=False)
    return hist


def final_loss(h): return h[-1][3]
def final_acc(h):  return h[-1][4]
def step_reaching(h, target):
    for step, _, _, vl, _ in h:
        if vl <= target:
            return step
    return None


def main():
    n_tot = len(G.train_data) + len(G.val_data)
    print(f"corpus={n_tot:,} chars  vocab={G.vocab_size}  C={G.C} T={G.CTX}"
          f"  |  {STEPS} steps, seeds {SEEDS}\n")

    makers = {
        "uniform":          lambda s: G.UniformSampler("train", B=32, seed=s),
        "prio (no IS)":     lambda s: G.PrioritizedSampler("train", B=32, seed=s,
                                          beta=0.9, eps=0.1, is_correct=False),
        "prio (IS-corr)":   lambda s: G.PrioritizedSampler("train", B=32, seed=s,
                                          beta=0.9, eps=0.1, is_correct=True),
    }

    res = {n: [run(mk, s) for s in SEEDS] for n, mk in makers.items()}
    uni_final = {i: final_loss(res["uniform"][i]) for i in range(len(SEEDS))}

    print(f"{'sampler':16} {'val loss (mean±std)':>22} {'val acc':>9} {'->match uni @':>14}")
    print("-" * 64)
    for name in makers:
        hs = res[name]
        losses = [final_loss(h) for h in hs]
        accs = [final_acc(h) for h in hs]
        if name == "uniform":
            sp = f"{STEPS} (ref)"
        else:
            st = [step_reaching(hs[i], uni_final[i]) for i in range(len(SEEDS))]
            st = [x for x in st if x is not None]
            sp = f"{np.mean(st):.0f}" if st else "never"
        print(f"{name:16} {np.mean(losses):10.3f} ± {np.std(losses):5.3f}"
              f"      {np.mean(accs):8.1%} {sp:>14}")


if __name__ == "__main__":
    main()
