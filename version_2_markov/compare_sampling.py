"""
Head-to-head: uniform vs prioritized ("Markov-over-positions") sampling,
now on a large, non-redundant corpus (corpus.txt) where there is real
headroom -- the model is nowhere near saturating, so a better sampler can show
up as better FINAL quality, not just faster early convergence.

Same model, seed/init, optimizer (Adam), step budget. Only the batch sampler
differs. Metric: val loss / accuracy over training (lower loss = better).

Run:  python compare_sampling.py
"""

import numpy as np
import step3_gpt as G


def best_val_loss(history):
    return min(va_loss for _, _, _, va_loss, _ in history)


def run(make_sampler, steps, seed, eval_every):
    sampler = make_sampler(seed)
    p, hist = G.train(steps=steps, B=32, seed=seed, sampler=sampler,
                      eval_every=eval_every, verbose=False)
    eval_rng = np.random.default_rng(123)
    va = [G.evaluate(p, "val", eval_rng) for _ in range(8)]
    return dict(hist=hist,
                va_loss=np.mean([l for l, _ in va]),
                va_acc=np.mean([a for _, a in va]),
                best=best_val_loss(hist))


def main(steps=2000, seeds=(0, 1), eval_every=100):
    print(f"corpus={len(G.train_data)+len(G.val_data):,} chars  vocab={G.vocab_size}"
          f"  C={G.C} T={G.BLOCK}  |  {steps} steps, seeds {seeds}\n")
    makers = {
        "uniform":     lambda s: G.UniformSampler("train", B=32, seed=s),
        "prioritized": lambda s: G.PrioritizedSampler("train", B=32, seed=s,
                                                       beta=0.9, eps=0.1),
    }

    # run each (sampler, seed) exactly once
    results = {n: [run(mk, steps, s, eval_every) for s in seeds]
               for n, mk in makers.items()}

    # seed-0 val-loss curve, side by side
    print(f"seed {seeds[0]} val loss / val acc:")
    print(f"{'step':>5} | {'uniform':>16} | {'prioritized':>16}")
    print("-" * 46)
    for (s, _, _, ul, ua), (_, _, _, pl, pa) in zip(results["uniform"][0]["hist"],
                                                     results["prioritized"][0]["hist"]):
        if s == 1 or s % 200 == 0:
            print(f"{s:>5} | {ul:7.3f} {ua:6.1%} | {pl:7.3f} {pa:6.1%}")

    # headline numbers averaged over seeds
    print(f"\naveraged over seeds {seeds}:")
    print(f"{'':12} {'final val loss':>15} {'final val acc':>14} {'best val loss':>14}")
    for name in makers:
        rs = results[name]
        vl = np.mean([r["va_loss"] for r in rs])
        va = np.mean([r["va_acc"] for r in rs])
        bl = np.mean([r["best"] for r in rs])
        print(f"{name:12} {vl:15.3f} {va:14.1%} {bl:14.3f}")


if __name__ == "__main__":
    main()
