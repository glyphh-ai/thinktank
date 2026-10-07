"""
Strat 0 — the new base must speak by step 5,000
===============================================
Train the 4-layer / 4-head / C=256 word-level GPT (5.0M params) for exactly
5,000 AdamW steps (B=32 x T=64 = 10.2M tokens, < 1 epoch of the 24M-token
train slice, so val loss is an honest generalization number).

Run:  python strat0_train.py            # ~1 h on a 4-core CPU
Out:  strat0_params.npz (checkpointed at every log), strat0_history.json,
      samples printed at the end
"""

import json

import numpy as np

import step3_gpt as G

STEPS = 5000
B = 32
LR = 2e-3

def checkpoint(step, p):
    np.savez("strat0_params.npz", step=step, **p)

if __name__ == "__main__":
    print(f"vocab={G.vocab_size}  L={G.N_LAYER}  H={G.N_HEAD}  C={G.C}  T={G.BLOCK}  "
          f"params={G.n_params(G.init_params(np.random.default_rng(0))):,}  "
          f"steps={STEPS}  B={B}  lr={LR}\n", flush=True)
    p, hist = G.train(steps=STEPS, B=B, lr=LR, log_every=250, on_log=checkpoint)

    with open("strat0_history.json", "w") as f:
        json.dump([dict(zip(["step", "train_loss", "train_acc", "val_loss", "val_acc"], h))
                   for h in hist], f, indent=1)
    final = G.evaluate(p, "val", np.random.default_rng(7), batches=40)
    print(f"\nfinal val (40 batches x 64): loss {final[0]:.3f}  acc {final[1]:.1%}")

    for seed in (0, 1, 2):
        print(f"\n--- sample seed={seed} (temperature=0.8, top_k=40) ---")
        print(G.sample(p, n=160, seed=seed))
