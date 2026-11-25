"""step2_transformer.py — the regular transformer on the same bytes.

A byte-level causal transformer (MLX, backprop, AdamW) trained on the same
4.0 MB of five novels and scored on the same Pride and Prejudice text as
step1, so bits/char compare directly. Weights go to runs/ for step3.

Run:  python step2_transformer.py [--steps 6000] [--d 384] [--layers 6]
"""

import argparse
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import step1_lm

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
CTX = 256


class ByteLM(nn.Module):
    def __init__(self, d=384, layers=6, heads=6, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(256, d)
        self.pos = nn.Embedding(CTX, d)
        self.enc = nn.TransformerEncoder(layers, d, heads, dropout=dropout, norm_first=True)
        self.out = nn.Linear(d, 256)

    def __call__(self, x):
        L = x.shape[1]
        h = self.emb(x) + self.pos(mx.arange(L))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        return self.out(self.enc(h, mask))


def params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))


def bits_per_char(model, data, stride=128):
    """Every byte scored once, each with at least CTX - stride bytes of context."""
    model.eval()
    x = np.frombuffer(data, np.uint8).astype(np.int32)
    total, scored = 0.0, 0
    starts = list(range(0, max(1, len(x) - CTX), stride))
    for k in range(0, len(starts), 64):
        s = starts[k:k + 64]
        win = np.stack([x[a:a + CTX + 1] for a in s if a + CTX + 1 <= len(x)])
        if not len(win):
            break
        logp = nn.log_softmax(model(mx.array(win[:, :-1])), axis=-1)
        tgt = mx.array(win[:, 1:])
        nll = -mx.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]
        nll = np.array(nll) / math.log(2)
        for row, a in zip(nll, s):
            keep = row if a == 0 else row[CTX - stride:]
            total += keep.sum()
            scored += len(keep)
    model.train()
    return total / scored


def loss_fn(model, xb, yb):
    return nn.losses.cross_entropy(model(xb), yb, reduction="mean")


def train(model, data, steps, batch=32, lr=1e-3, seed=0, log_every=500):
    x = np.frombuffer(data, np.uint8).astype(np.int32)
    rng = np.random.default_rng(seed)
    sched = optim.join_schedules([optim.linear_schedule(1e-5, lr, 200),
                                  optim.cosine_decay(lr, steps - 200, lr * 0.05)], [200])
    opt = optim.AdamW(sched, weight_decay=0.05)
    step = nn.value_and_grad(model, loss_fn)
    t0 = time.time()
    for i in range(steps):
        a = rng.integers(0, len(x) - CTX - 1, batch)
        win = np.stack([x[j:j + CTX + 1] for j in a])
        loss, g = step(model, mx.array(win[:, :-1]), mx.array(win[:, 1:]))
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state)
        if (i + 1) % log_every == 0:
            print(f"  step {i + 1:>5}  loss {loss.item() / math.log(2):.3f} bits  "
                  f"{time.time() - t0:.0f}s", flush=True)


def main(steps, d, layers):
    train_data, test = step1_lm.corpus(), step1_lm.test_text(200_000)
    mx.random.seed(0)
    model = ByteLM(d, layers, heads=d // 64)
    n = params(model)
    print(f"transformer d={d} L={layers}: {n / 1e6:.1f}M params ({n * 4 / 1e6:.0f} MB fp32), "
          f"ctx {CTX}, train {len(train_data) / 1e6:.2f} MB")
    t = time.time()
    train(model, train_data, steps)
    bpc = bits_per_char(model, test)
    print(f"\ntransformer frozen {bpc:.3f} bits/char on Pride and Prejudice "
          f"({len(test):,} chars, train {time.time() - t:.0f}s)")
    os.makedirs(RUNS, exist_ok=True)
    model.save_weights(os.path.join(RUNS, f"byte_lm_d{d}_L{layers}.safetensors"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    a = ap.parse_args()
    main(a.steps, a.d, a.layers)
