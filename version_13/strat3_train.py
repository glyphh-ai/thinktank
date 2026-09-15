"""
Strat 3 — HYBRID: identity + authored semantics
===============================================
Strat 2's earned failure: the lattice is a category code — 243 words can
share one cortex vector, so the model loses word identity and even random
unique vectors beat it. Fix: give every word a name tag AND the geometry.

    E[word] = row_norm( alpha * lattice_proj[word] + beta * id[word] )

id[word] is a deterministic unique unit vector (atom seeded by the word
string). Two mixes, same data/seeds/steps as strat 2; random + learned
numbers are reused from strat2_results.json (identical protocol).

Pre-registered gates:
    G1'  best hybrid beats frozen-random by >= 0.05 val loss
         -> semantics ADDS value once identity is restored
    G2'  best hybrid within 0.15 of learned -> authored table can replace
         the learned embedding layer

Run:  python -u strat3_train.py         # ~30 min CPU
Out:  strat3_results.json, strat3_params_<mix>.npz
"""

import json

import numpy as np

import step3_gpt as G
from step1_data import words as VOCAB
from strat1_vsa import atom
from strat2_train import STEPS, B, LR, lattice_table, _row_norm

MIXES = [("a10_b05", 1.0, 0.5), ("a05_b10", 0.5, 1.0)]

def id_table() -> np.ndarray:
    """Unique deterministic unit vector per word — the name tag."""
    E = np.stack([atom(f"id:{w}")[: G.C].astype(np.float64) for w in VOCAB])
    return _row_norm(E)

if __name__ == "__main__":
    prior = json.load(open("strat2_results.json"))
    lat, ids = lattice_table(), id_table()

    results = {"random": prior["random"], "learned": prior["learned"]}
    for name, a, b_ in MIXES:
        print(f"=== hybrid {name}: E = {a}*lattice + {b_}*id ===", flush=True)
        table = _row_norm(a * lat + b_ * ids)
        p, hist = G.train(steps=STEPS, B=B, lr=LR, log_every=1000,
                          tok_emb=table)
        va_loss = float(np.mean([h[3] for h in hist[-3:]]))
        va_acc = float(np.mean([h[4] for h in hist[-3:]]))
        results[name] = {"val_loss": va_loss, "val_acc": va_acc}
        np.savez(f"strat3_params_{name}.npz", **p)
        print(f"--- {name}: val loss {va_loss:.3f}  acc {va_acc:.1%}")
        print(G.sample(p, n=60, seed=42, temperature=0.7), "\n", flush=True)

    best = min(MIXES, key=lambda m: results[m[0]]["val_loss"])[0]
    g1 = results["random"]["val_loss"] - results[best]["val_loss"]
    g2 = results[best]["val_loss"] - results["learned"]["val_loss"]
    results["gates"] = {
        "best_mix": best,
        "G1_hybrid_beats_random_by": round(g1, 4), "G1_pass": g1 >= 0.05,
        "G2_hybrid_gap_to_learned": round(g2, 4), "G2_pass": g2 <= 0.15,
    }
    json.dump(results, open("strat3_results.json", "w"), indent=2)

    print("=" * 60)
    for k in [m[0] for m in MIXES] + ["random", "learned"]:
        print(f"{k:<10} val loss {results[k]['val_loss']:.3f}  "
              f"acc {results[k]['val_acc']:.1%}")
    print(f"G1' {best} > random by {g1:+.3f}  "
          f"({'PASS' if results['gates']['G1_pass'] else 'FAIL'}, need >= +0.05)")
    print(f"G2' {best} - learned  {g2:+.3f}  "
          f"({'PASS' if results['gates']['G2_pass'] else 'FAIL'}, need <= +0.15)")
