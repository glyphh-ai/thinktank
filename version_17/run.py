"""
Run — the pre-registered sweep
==============================
4 models x 4 training-set sizes x 3 seeds. Train on depths 1-4 only,
evaluate on fresh circuits at depths 1..32. Writes results.json and
prints the gate table.

Run:  python -u run.py            (python -u run.py --smoke  for a timing check)
"""

import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from circuits import DEEP_GATES, make_set
from models import build, random_bank, train_bank

MODELS = ["A", "B", "C", "C0"]
SIZES = [64, 256, 1024, 8192]
SEEDS = [0, 1, 2]
TRAIN_DEPTHS = [1, 2, 3, 4]
EVAL_DEPTHS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
EVAL_N = 1000
STEPS = 3000
BATCH = 64
LR = {"A": 1e-3, "B": 1e-2, "C": 1e-2, "C0": 1e-2}


def tensors(s):
    return [torch.from_numpy(np.asarray(a)) for a in s]


def train_set(n, seed):
    rng = np.random.default_rng(1000 + seed)
    return {d: tensors(make_set(n // len(TRAIN_DEPTHS), d, rng)) for d in TRAIN_DEPTHS}


def eval_sets():
    rng = np.random.default_rng(99)
    return {d: tensors(make_set(EVAL_N, d, rng)) for d in EVAL_DEPTHS}


def fit(model, data, lr, seed, steps):
    rng = np.random.default_rng(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        x, ops, addr, y = data[TRAIN_DEPTHS[rng.integers(len(TRAIN_DEPTHS))]]
        ix = torch.from_numpy(rng.integers(0, len(y), BATCH))
        p = model(x[ix], ops[ix], addr[ix]).clamp(1e-6, 1 - 1e-6)
        loss = F.binary_cross_entropy(p, y[ix].float())
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def accuracy(model, evals):
    return {d: float(((model(x, o, a) > 0.5).long() == y).float().mean())
            for d, (x, o, a, y) in evals.items()}


def router_hits(model):
    """How many ops does C's router send to the tiny model of that exact gate?"""
    best = model.router.argmax(-1).tolist()
    return sum(best[i] == g for i, g in enumerate(DEEP_GATES))


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    steps = 100 if smoke else STEPS
    bank = train_bank()
    with torch.no_grad():
        a = torch.tensor([0., 0., 1., 1.]); b = torch.tensor([0., 1., 0., 1.])
        from circuits import TABLE
        err = (bank(a, b) - torch.tensor(TABLE.T)).abs().max().item()
    print(f"bank: 16 tiny models, max |error| on their truth tables = {err:.2e}")

    evals = eval_sets()
    results = []
    for n in SIZES:
        for seed in SEEDS:
            data = train_set(n, seed)
            for name in MODELS:
                model = build(name, seed, bank if name == "C" else random_bank(seed))
                t0 = time.time()
                fit(model, data, LR[name], seed, steps)
                acc = accuracy(model, evals)
                row = dict(model=name, n=n, seed=seed, acc=acc,
                           n_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                           secs=round(time.time() - t0, 1))
                if name in ("C", "C0"):
                    row["router_hits"] = router_hits(model)
                results.append(row)
                print(f"{name:2s} n={n:5d} seed={seed} | ID {np.mean([acc[d] for d in TRAIN_DEPTHS]):.3f}"
                      f" | d16 {acc[16]:.3f} | d32 {acc[32]:.3f}"
                      f"{' | router ' + str(row['router_hits']) + '/10' if 'router_hits' in row else ''}"
                      f" | {row['secs']}s", flush=True)
            if smoke:
                break
        if smoke:
            break

    if not smoke:
        with open("results.json", "w") as f:
            json.dump(results, f, indent=1)
        import report
        report.main()
