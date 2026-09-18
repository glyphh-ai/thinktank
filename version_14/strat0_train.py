"""
Strat 0 — the baseline must speak
=================================
Train the from-scratch word-level GPT on TinyStories until it produces
non-gibberish. No lattice, no primes — this run only earns the substrate:
a corpus + model pair strong enough that later semantic interventions have
something real to move.

Run:  python strat0_train.py            # ~15 min CPU
Out:  strat0_params.npz, samples printed at the end
"""

import numpy as np

import step3_gpt as G

STEPS = 8000
B = 32
LR = 2e-3

if __name__ == "__main__":
    print(f"vocab={G.vocab_size}  C={G.C}  T={G.BLOCK}  "
          f"params={sum(v.size for v in G.init_params(np.random.default_rng(0)).values()):,}")
    p, hist = G.train(steps=STEPS, B=B, lr=LR, log_every=500)

    np.savez("strat0_params.npz", **p)
    print("\nsaved -> strat0_params.npz")

    for temp in (0.7, 0.9):
        print(f"\n--- sample (temperature={temp}) ---")
        print(G.sample(p, n=150, seed=42, temperature=temp))
