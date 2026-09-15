"""
Strat 4 — LEARNED GAIN: one auditable scalar per word
=====================================================
Strat 3's hybrid still trails learned embeddings by 0.29. One suspected
channel: learned embeddings encode word FREQUENCY in their norms; our
row-normalized frozen table erases it. Restore exactly that channel and
nothing else:

    E[word] = s_w * row_norm( 1.0 * lattice_proj + 0.5 * id )

The table stays frozen; only s (one scalar per word, init 1.0) learns —
5,375 parameters, each readable as "this word's loudness". The semantic
DIRECTIONS remain 100% authored.

Pre-registered gates:
    G1''  gain-hybrid improves on strat 3's best (3.526) by >= 0.05
    G2''  gain-hybrid within 0.15 of learned (3.235)

Run:  python -u strat4_train.py         # ~15 min CPU
Out:  strat4_results.json, strat4_params.npz
"""

import json

import numpy as np

import step3_gpt as G
from step1_data import itos
from strat2_train import STEPS, B, LR, lattice_table, _row_norm
from strat3_train import id_table

if __name__ == "__main__":
    prior = json.load(open("strat3_results.json"))
    table = _row_norm(1.0 * lattice_table() + 0.5 * id_table())

    print("=== gain-hybrid: frozen table + learned per-word scalar ===",
          flush=True)
    p, hist = G.train(steps=STEPS, B=B, lr=LR, log_every=1000,
                      tok_emb=table, tok_gain=True)
    va_loss = float(np.mean([h[3] for h in hist[-3:]]))
    va_acc = float(np.mean([h[4] for h in hist[-3:]]))
    np.savez("strat4_params.npz", **p)
    print(f"--- gain_hybrid: val loss {va_loss:.3f}  acc {va_acc:.1%}")
    print(G.sample(p, n=60, seed=42, temperature=0.7), "\n", flush=True)

    # the gain vector is itself a finding: what did the corpus turn up/down?
    s = p["tok_gain"]
    order = np.argsort(s)
    print("quietest words:", [(itos[int(i)], round(float(s[i]), 2))
                              for i in order[:8]])
    print("loudest words :", [(itos[int(i)], round(float(s[i]), 2))
                              for i in order[-8:]])

    best3 = prior[prior["gates"]["best_mix"]]["val_loss"]
    learned = prior["learned"]["val_loss"]
    g1 = best3 - va_loss
    g2 = va_loss - learned
    results = {"gain_hybrid": {"val_loss": va_loss, "val_acc": va_acc},
               "strat3_best": best3, "learned": learned,
               "gates": {"G1_improves_strat3_by": round(g1, 4),
                         "G1_pass": g1 >= 0.05,
                         "G2_gap_to_learned": round(g2, 4),
                         "G2_pass": g2 <= 0.15}}
    json.dump(results, open("strat4_results.json", "w"), indent=2)

    print("=" * 60)
    print(f"gain_hybrid val loss {va_loss:.3f}  acc {va_acc:.1%}")
    print(f"G1'' improves strat3 best by {g1:+.3f}  "
          f"({'PASS' if g1 >= 0.05 else 'FAIL'}, need >= +0.05)")
    print(f"G2'' gap to learned {g2:+.3f}  "
          f"({'PASS' if g2 <= 0.15 else 'FAIL'}, need <= +0.15)")
