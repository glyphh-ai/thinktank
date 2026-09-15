"""
Strat 2 — THE GATE: can the model live in the authored geometry?
================================================================
Three trainings of the SAME model on the SAME data, seeds, and steps.
The only variable is the input embedding table:

    lattice : FROZEN — the 5,374 Haiku-judged cortex vectors, randomly
              projected 100k -> C and row-normalized
    random  : FROZEN — random unit vectors, same norm treatment (control)
    learned : free tok_emb (the strat-0 baseline condition)

Pre-registered gates (design committed before results):
    G1  lattice beats random by >= 0.10 val loss  -> the structure is real
        signal to a transformer, not just "any fixed basis works"
    G2  lattice reaches within 0.15 val loss of learned -> authored
        semantics can REPLACE the learned embedding layer (the headline)

Run:  python strat2_train.py            # ~45 min CPU
Out:  strat2_results.json, strat2_params_<cond>.npz, samples per condition
"""

import json

import numpy as np

import step3_gpt as G
from strat1_encode import load_bank
from strat1_vsa import atom, D

STEPS = 8000
B = 32
LR = 2e-3

def projection(seed_name="strat2:proj", C=G.C) -> np.ndarray:
    """Deterministic Johnson-Lindenstrauss projection (D, C)."""
    seed = int.from_bytes(__import__("hashlib").sha256(seed_name.encode())
                          .digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1.0 / np.sqrt(C), size=(D, C))

def lattice_table() -> np.ndarray:
    bank = load_bank()
    E = bank["cortex"].astype(np.float64) @ projection()
    # unjudged rows (just <unk>) get a deterministic random vector so the
    # token still has an identity
    for i in np.where(~bank["judged"])[0]:
        E[i] = atom(f"unjudged:{i}")[: G.C].astype(np.float64)
    return _row_norm(E)

def random_table() -> np.ndarray:
    rng = np.random.default_rng(1234)
    return _row_norm(rng.normal(0, 1, size=(G.vocab_size, G.C)))

def _row_norm(E: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(E, axis=1, keepdims=True)
    n[n == 0] = 1
    return E / n

def check_projection(E_full: np.ndarray, E_proj: np.ndarray, n=2000):
    """How well does the C-dim projection preserve 100k-D cosine?"""
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(E_full), size=(n, 2))
    def cos(M, i, j):
        a, b = M[i], M[j]
        return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    full = np.array([cos(E_full, i, j) for i, j in idx])
    proj = np.array([cos(E_proj, i, j) for i, j in idx])
    return float(np.corrcoef(full, proj)[0, 1])

if __name__ == "__main__":
    bank = load_bank()
    lat = lattice_table()
    r = check_projection(bank["cortex"].astype(np.float64), lat)
    print(f"projection 100k->{G.C}: cosine preservation r={r:.4f}\n")

    conditions = {
        "lattice": lat,
        "random": random_table(),
        "learned": None,
    }
    results = {}
    for name, table in conditions.items():
        print(f"=== condition: {name} ===")
        p, hist = G.train(steps=STEPS, B=B, lr=LR, log_every=1000,
                          tok_emb=table)
        va_loss = float(np.mean([h[3] for h in hist[-3:]]))
        va_acc = float(np.mean([h[4] for h in hist[-3:]]))
        results[name] = {"val_loss": va_loss, "val_acc": va_acc}
        np.savez(f"strat2_params_{name}.npz", **p)
        print(f"--- {name}: val loss {va_loss:.3f}  acc {va_acc:.1%}")
        print(G.sample(p, n=60, seed=42, temperature=0.7), "\n")

    g1 = results["random"]["val_loss"] - results["lattice"]["val_loss"]
    g2 = results["lattice"]["val_loss"] - results["learned"]["val_loss"]
    results["gates"] = {
        "G1_lattice_beats_random_by": round(g1, 4), "G1_pass": g1 >= 0.10,
        "G2_lattice_gap_to_learned": round(g2, 4), "G2_pass": g2 <= 0.15,
    }
    json.dump(results, open("strat2_results.json", "w"), indent=2)

    print("=" * 60)
    for n_ in ("lattice", "random", "learned"):
        print(f"{n_:<8} val loss {results[n_]['val_loss']:.3f}  "
              f"acc {results[n_]['val_acc']:.1%}")
    print(f"G1 lattice > random by {g1:+.3f}  "
          f"({'PASS' if results['gates']['G1_pass'] else 'FAIL'}, need >= +0.10)")
    print(f"G2 lattice - learned  {g2:+.3f}  "
          f"({'PASS' if results['gates']['G2_pass'] else 'FAIL'}, need <= +0.15)")
