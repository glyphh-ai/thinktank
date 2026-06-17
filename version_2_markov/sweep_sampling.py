"""
Rigor pass for the sampling experiment: multiple seeds + a beta/eps grid, with
error bars and a train-efficiency metric.

For each config we report final val loss/acc as mean +/- std over seeds, and
"speedup": for prioritized, the earliest step it reaches the SAME-seed uniform
run's final val loss (i.e. how many fewer steps to match uniform's end state).

Run:  python sweep_sampling.py
"""

import numpy as np
import step3_gpt as G

STEPS = 2000
SEEDS = (0, 1, 2)
EVAL_EVERY = 100


def train_once(make_sampler, seed):
    sampler = make_sampler(seed)
    _, hist = G.train(steps=STEPS, B=32, seed=seed, sampler=sampler,
                      eval_every=EVAL_EVERY, verbose=False)
    return hist


def final_val_loss(hist):
    return hist[-1][3]

def final_val_acc(hist):
    return hist[-1][4]

def step_reaching(hist, target_loss):
    """First step whose val loss <= target (None if never)."""
    for step, _, _, va_loss, _ in hist:
        if va_loss <= target_loss:
            return step
    return None


def main():
    n_tot = len(G.train_data) + len(G.val_data)
    print(f"corpus={n_tot:,} chars  vocab={G.vocab_size}  C={G.C} T={G.BLOCK}"
          f"  |  {STEPS} steps, seeds {SEEDS}\n")

    # baseline: uniform, per seed (needed as the target for speedup)
    uni = {s: train_once(lambda sd: G.UniformSampler("train", B=32, seed=sd), s)
           for s in SEEDS}
    uni_final = {s: final_val_loss(uni[s]) for s in SEEDS}

    configs = {
        "uniform": None,
        "prio b=.9 e=.10":  dict(beta=0.9,  eps=0.10),
        "prio b=.99 e=.10": dict(beta=0.99, eps=0.10),
        "prio b=.9 e=.05":  dict(beta=0.9,  eps=0.05),
        "prio b=.9 e=.20":  dict(beta=0.9,  eps=0.20),
    }

    print(f"{'config':18} {'val loss (mean±std)':>22} {'val acc':>9} {'->match uni @':>14}")
    print("-" * 66)
    for name, kw in configs.items():
        losses, accs, speed = [], [], []
        for s in SEEDS:
            if kw is None:
                hist = uni[s]
            else:
                hist = train_once(
                    lambda sd, kw=kw: G.PrioritizedSampler("train", B=32, seed=sd, **kw), s)
            losses.append(final_val_loss(hist))
            accs.append(final_val_acc(hist))
            if kw is not None:
                st = step_reaching(hist, uni_final[s])   # match same-seed uniform end
                if st is not None:
                    speed.append(st)
        l_m, l_s = np.mean(losses), np.std(losses)
        a_m = np.mean(accs)
        if kw is None:
            sp = f"{STEPS} (ref)"
        else:
            sp = f"{np.mean(speed):.0f}" if speed else "never"
        print(f"{name:18} {l_m:10.3f} ± {l_s:5.3f}      {a_m:8.1%} {sp:>14}")


if __name__ == "__main__":
    main()
