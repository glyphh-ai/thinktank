"""
Head-to-head: Adam vs MarkovOpt (sign-Markov adaptive stepper).

Same model, same seed/init, same batches, same step budget. We compare the
learning curves, the final loss/accuracy, and how fast each reaches 90% val
accuracy. This is the experiment that tells us whether the probability layer
actually buys anything over Adam.

Run:  python compare_optimizers.py
"""

import numpy as np
import step3_gpt as G


def steps_to(history, target_acc=0.90):
    """First logged step whose val accuracy >= target (None if never)."""
    for step, _, _, _, va_acc in history:
        if va_acc >= target_acc:
            return step
    return None


def run(make_opt, steps, seed):
    p, hist = G.train(steps=steps, B=32, seed=seed, make_opt=make_opt,
                      eval_every=25, verbose=False)
    eval_rng = np.random.default_rng(123)
    tr = [G.evaluate(p, "train", eval_rng) for _ in range(8)]
    va = [G.evaluate(p, "val", eval_rng) for _ in range(8)]
    return dict(
        hist=hist,
        tr_loss=np.mean([l for l, _ in tr]), tr_acc=np.mean([a for _, a in tr]),
        va_loss=np.mean([l for l, _ in va]), va_acc=np.mean([a for _, a in va]),
        hit90=steps_to(hist, 0.90),
    )


def main(steps=700, seed=0):
    print(f"model: vocab={G.vocab_size} C={G.C} T={G.BLOCK}  |  {steps} steps, seed {seed}\n")

    opts = {
        "Adam":       lambda p: G.Adam(p, lr=3e-3),
        "MarkovOpt":  lambda p: G.MarkovOpt(p, lr=5e-3),
        "MarkovAdam": lambda p: G.MarkovAdam(p, lr=3e-3),
    }
    results = {name: run(mk, steps, seed) for name, mk in opts.items()}

    # learning curves side by side
    names = list(opts)
    print(f"{'step':>5} | " + " | ".join(f"{n+' loss/acc':>18}" for n in names))
    print("-" * (8 + 21 * len(names)))
    hs = [results[n]["hist"] for n in names]
    for i, row in enumerate(zip(*hs)):
        s = row[0][0]
        if s == 1 or s % 100 == 0:
            cells = " | ".join(f"{r[3]:7.3f} {r[4]:6.1%}" for r in row)
            print(f"{s:>5} | {cells}")

    # summary
    width = 56
    print("\n" + "=" * width)
    print(f"{'':12} {'val loss':>9} {'val acc':>9} {'train acc':>10} {'->90% @':>9}")
    for name in names:
        r = results[name]
        hit = f"{r['hit90']}" if r["hit90"] else "never"
        print(f"{name:12} {r['va_loss']:9.3f} {r['va_acc']:9.1%}"
              f" {r['tr_acc']:10.1%} {hit:>9}")
    print("=" * width)


if __name__ == "__main__":
    main()
