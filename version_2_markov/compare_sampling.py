"""
Head-to-head: uniform vs prioritized ("Markov-over-positions") sampling.

Same model, same seed/init, same optimizer (Adam), same step budget. The only
difference is HOW each batch's windows are chosen:

  - UniformSampler:     B windows drawn uniformly at random (the baseline).
  - PrioritizedSampler: P(draw window) tracks an EMA of the loss the model
                        recently got there -> the batch concentrates on the
                        windows it's still weak on.

Metric that matters here: steps-to-90% val accuracy (does it learn FASTER?),
plus final val loss/acc (did focusing on hard windows hurt generalization?).

Run:  python compare_sampling.py
"""

import numpy as np
import step3_gpt as G


def steps_to(history, target=0.90):
    for step, _, _, _, va_acc in history:
        if va_acc >= target:
            return step
    return None


def run(make_sampler, steps, seed):
    sampler = make_sampler(seed)
    p, hist = G.train(steps=steps, B=32, seed=seed, sampler=sampler,
                      eval_every=25, verbose=False)
    eval_rng = np.random.default_rng(123)
    va = [G.evaluate(p, "val", eval_rng) for _ in range(8)]
    tr = [G.evaluate(p, "train", eval_rng) for _ in range(8)]
    return dict(hist=hist,
                va_loss=np.mean([l for l, _ in va]), va_acc=np.mean([a for _, a in va]),
                tr_acc=np.mean([a for _, a in tr]),
                hit90=steps_to(hist, 0.90), hit95=steps_to(hist, 0.95))


def main(steps=700, seeds=(0, 1, 2)):
    print(f"model: vocab={G.vocab_size} C={G.C} T={G.BLOCK}  |  {steps} steps,"
          f" seeds {seeds}\n")
    makers = {
        "uniform":     lambda s: G.UniformSampler("train", B=32, seed=s),
        "prioritized": lambda s: G.PrioritizedSampler("train", B=32, seed=s,
                                                       beta=0.9, eps=0.1),
    }

    # one detailed curve on the first seed
    print(f"seed {seeds[0]} learning curves:")
    print(f"{'step':>5} | {'uniform loss/acc':>18} | {'prioritized loss/acc':>20}")
    print("-" * 52)
    curves = {n: run(mk, steps, seeds[0]) for n, mk in makers.items()}
    for (s, _, _, ul, ua), (_, _, _, pl, pa) in zip(curves["uniform"]["hist"],
                                                     curves["prioritized"]["hist"]):
        if s == 1 or s % 100 == 0:
            print(f"{s:>5} | {ul:8.3f} {ua:6.1%}     | {pl:8.3f} {pa:6.1%}")

    # average the headline numbers over seeds (small corpus -> noisy single runs)
    print(f"\naveraged over seeds {seeds}:")
    print(f"{'':12} {'val loss':>9} {'val acc':>9} {'->90% @':>9} {'->95% @':>9}")
    for name, mk in makers.items():
        rs = [run(mk, steps, s) for s in seeds]
        vl = np.mean([r["va_loss"] for r in rs])
        va = np.mean([r["va_acc"] for r in rs])
        h90 = [r["hit90"] for r in rs if r["hit90"]]
        h95 = [r["hit95"] for r in rs if r["hit95"]]
        s90 = f"{np.mean(h90):.0f}" if h90 else "never"
        s95 = f"{np.mean(h95):.0f}" if h95 else "never"
        print(f"{name:12} {vl:9.3f} {va:9.1%} {s90:>9} {s95:>9}")


if __name__ == "__main__":
    main()
