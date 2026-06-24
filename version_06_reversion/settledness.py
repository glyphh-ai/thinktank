"""
settledness.py — is the Markov stay-probability r a real "settledness" signal?

r[k] (per weight) is the EMA of gradient sign-agreement that Adam(reversion=True)
already tracks. Hypothesis: a high r means that weight has stopped thrashing and
"settled". If that's true, it should be a usable diagnostic:

    freezing the MOST-settled weights should hurt final loss LESS
    than freezing the same number of LEAST-settled weights.

If the two freeze arms are indistinguishable, r is not telling us anything that
generalises -- it's just optimizer bookkeeping. This script runs that test and
also prints per-layer settledness so we can see what settles first.

Run:  python settledness.py
"""

import numpy as np
import step3_gpt as G

WARMUP = 150          # steps before we snapshot r and start freezing
TOTAL = 700
FREEZE_FRAC = 0.5     # freeze this fraction of all weights
SEEDS = range(4)


def train_with_freeze(seed, frozen_select=None):
    """Train TOTAL steps. After WARMUP, freeze a set of weights chosen from the
    settledness snapshot. frozen_select(r_dict) -> {k: boolean mask} or None."""
    rng = np.random.default_rng(seed)
    p = G.init_params(rng)
    opt = G.Adam(p, lr=3e-3, reversion=True, alpha=1.0)
    masks = None
    for step in range(1, TOTAL + 1):
        x, y = G.get_batch("train", rng)
        _, _, cache = G.forward(p, x, y)
        grads = G.backward(p, cache)
        if step == WARMUP and frozen_select is not None:
            masks = frozen_select(opt.r)
        if masks is not None:
            for k in grads:
                grads[k] = grads[k] * (1.0 - masks[k])  # zero frozen weights' grad
        opt.step(p, grads)
    eval_rng = np.random.default_rng(999)
    return G.evaluate(p, "val", eval_rng), opt.r


def per_layer_settledness(r):
    return {k: float(v.mean()) for k, v in r.items()}


def global_threshold_masks(r, freeze_settled):
    """Mask the FREEZE_FRAC most- (or least-) settled weights, network-wide."""
    allvals = np.concatenate([v.ravel() for v in r.values()])
    if freeze_settled:                       # freeze high r -> threshold at (1-frac) quantile
        thr = np.quantile(allvals, 1.0 - FREEZE_FRAC)
        return {k: (v >= thr).astype(float) for k, v in r.items()}
    else:                                    # freeze low r
        thr = np.quantile(allvals, FREEZE_FRAC)
        return {k: (v <= thr).astype(float) for k, v in r.items()}


if __name__ == "__main__":
    print(f"warmup={WARMUP}  total={TOTAL}  freeze_frac={FREEZE_FRAC}  seeds={list(SEEDS)}\n")

    # 1) what does the settledness landscape look like? (seed 0)
    (_, _), r0 = train_with_freeze(0, frozen_select=None)
    ls = per_layer_settledness(r0)
    print("per-layer mean settledness r at end of training (seed 0):")
    for k, v in sorted(ls.items(), key=lambda kv: -kv[1]):
        print(f"  {k:8s} {v:.3f}  {'#' * int(v * 40)}")
    print()

    # 2) falsification test: freeze settled vs freeze thrashing vs control
    ctrl, fset, fthr = [], [], []
    for s in SEEDS:
        (c_loss, c_acc), _ = train_with_freeze(s, frozen_select=None)
        (s_loss, s_acc), _ = train_with_freeze(
            s, frozen_select=lambda r: global_threshold_masks(r, freeze_settled=True))
        (t_loss, t_acc), _ = train_with_freeze(
            s, frozen_select=lambda r: global_threshold_masks(r, freeze_settled=False))
        ctrl.append((c_loss, c_acc)); fset.append((s_loss, s_acc)); fthr.append((t_loss, t_acc))
    ctrl, fset, fthr = map(lambda a: np.array(a), (ctrl, fset, fthr))

    def line(name, a):
        print(f"  {name:28s} val loss {a[:,0].mean():.3f}  acc {a[:,1].mean():5.1%}")
    print(f"after freezing {FREEZE_FRAC:.0%} of weights at step {WARMUP} (mean over {len(list(SEEDS))} seeds):")
    line("control (freeze nothing)", ctrl)
    line("freeze MOST-settled", fset)
    line("freeze LEAST-settled", fthr)
    print()
    print(f"freezing settled costs {fset[:,0].mean()-ctrl[:,0].mean():+.3f} val loss vs control")
    print(f"freezing thrashing costs {fthr[:,0].mean()-ctrl[:,0].mean():+.3f} val loss vs control")
    gap = fthr[:,0].mean() - fset[:,0].mean()
    print(f"\nsettled-minus-thrashing loss gap: {gap:+.3f}  "
          f"-> {'r IS a real settledness signal' if gap > 0.02 else 'r is NOT predictive (gap ~ 0)'}")
