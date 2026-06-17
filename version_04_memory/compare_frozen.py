"""
Trainable embedding ALONE vs trainable + a FROZEN meaning channel.

Both share identical vocab slots. The original-loop embedding trains as usual;
the frozen VSA meaning vector is ADDED in the same slot and never updates, so
the meaning substrate persists (can't wash out) while the trainable weights
learn against it.

    representation[id] = tok_emb[id] (trainable)  +  frozen_meaning[id] (fixed)

Run:  python compare_frozen.py
"""

import numpy as np
import step3_gpt as G
import step1_data as D
from train_meaning import vsa_embedding, steps_to


def run(frozen, seed, steps, eval_every):
    p, hist = G.train(steps=steps, B=32, seed=seed, frozen_tok=frozen,
                      eval_every=eval_every, verbose=False)
    eval_rng = np.random.default_rng(123)
    va = [G.evaluate(p, "val", eval_rng) for _ in range(8)]
    return p, hist, np.mean([l for l, _ in va]), np.mean([a for _, a in va])


def main(steps=600, seeds=(0, 1, 2), eval_every=25):
    C = G.C
    E = vsa_embedding(C)                       # (vocab, C) frozen meaning, id-ordered
    print(f"vocab={G.vocab_size} C={C} T={G.BLOCK} | {steps} steps, seeds {seeds}")
    print(f"frozen meaning channel: {E.shape}\n")

    configs = [("trainable only", None), ("trainable + frozen meaning", E)]
    print(f"{'config':28} {'val loss(mean±std)':>20} {'val acc':>9} {'->90% @':>9}")
    for name, frozen in configs:
        vls, vas, hit = [], [], []
        for s in seeds:
            _, h, vl, va = run(frozen, s, steps, eval_every)
            vls.append(vl); vas.append(va)
            st = steps_to(h, 0.90)
            if st:
                hit.append(st)
        hh = f"{np.mean(hit):.0f}" if hit else "never"
        print(f"{name:28} {np.mean(vls):10.3f} ± {np.std(vls):5.3f}   "
              f"{np.mean(vas):8.1%} {hh:>9}")
    # clear the global so later imports/tests see a clean model
    G.set_frozen_tok(None)


if __name__ == "__main__":
    main()
