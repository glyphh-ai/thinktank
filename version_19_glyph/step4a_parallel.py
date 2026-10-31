"""step4a_parallel.py — how dense can a glyph be if it's read in one glance?

Train a writer/reader pair (glyph_net.py) per patch size and measure exact
recall on held-out Pride and Prejudice. Chunks are L=64 chars; the patch
size P sets the density:

  P=8 -> 1.0 char/px   P=4 -> 4.0   P=3 -> 7.1   P=2 -> 16.0

The serial arithmetic code (step2b) reaches ~20 chars/px but reads one token
at a time. The prediction under test: a parallel read tops out near
4 chars/px.

Run:  .venv/bin/python step4a_parallel.py [--steps 4000] [--sigma 0]
"""

import argparse
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import glyph_net as gn
from step2_codec import HERE, TEST, TRAIN, book

L = 64


def windows(arr, n, rng):
    s = rng.integers(0, len(arr) - L, n)
    return arr[s[:, None] + np.arange(L)]


def run(P, steps, sigma, train_ids, test_ids, batch=256):
    mx.random.seed(0)
    rng = np.random.default_rng(0)
    w, r = gn.Writer(L, P), gn.Reader(L, P)

    class Pair(nn.Module):
        def __init__(self):
            super().__init__()
            self.w, self.r = w, r

    pair = Pair()
    warm = min(200, steps // 4)
    sched = optim.join_schedules([optim.linear_schedule(1e-5, 5e-4, warm),
                                  optim.cosine_decay(5e-4, steps - warm, 1e-5)], [warm])
    opt = optim.AdamW(sched, weight_decay=0.01)

    def loss_fn(m, x):
        logits = m.r(gn.channel(m.w(x), sigma, True))
        return nn.losses.cross_entropy(logits, x).mean()

    step = nn.value_and_grad(pair, loss_fn)
    t = time.time()
    for i in range(steps):
        x = mx.array(windows(train_ids, batch, rng))
        loss, g = step(pair, x)
        opt.update(pair, g)
        mx.eval(pair.parameters(), opt.state)

    # held-out eval through true 8-bit pixels
    x = windows(test_ids, 4096, np.random.default_rng(1))
    img = gn.channel(w(mx.array(x)), sigma, False)
    mx.eval(img)
    t0 = time.time()
    pred = np.array(mx.argmax(r(img), -1))
    read_s = time.time() - t0
    char_acc = (pred == x).mean()
    exact = (pred == x).all(1).mean()
    gn.save(os.path.join(HERE, "data", f"glyphnet_P{P}_s{sigma:g}"), w, r)
    sample = gn.text(pred[0]), gn.text(x[0])
    return dict(P=P, cpp=L / (P * P), char_acc=char_acc, exact=exact,
                train_s=time.time() - t, read_cps=x.size / read_s, sample=sample,
                loss=float(loss))


def main(steps, sigma, sizes):
    train_ids = gn.ids("\n\n".join(book(k, v) for k, v in TRAIN.items()))
    test_ids = gn.ids(book("pride_and_prejudice", TEST["pride_and_prejudice"]))
    print(f"L={L} chars per patch, channel 8-bit + noise sigma={sigma}, {steps} steps\n")
    print(f"{'P':>3}{'chars/px':>10}{'char acc':>10}{'chunk exact':>13}"
          f"{'read chars/s':>14}{'train':>8}")
    for P in sizes:
        m = run(P, steps, sigma, train_ids, test_ids)
        print(f"{m['P']:>3}{m['cpp']:>10.1f}{m['char_acc']:>10.4f}{m['exact']:>13.4f}"
              f"{m['read_cps']:>14,.0f}{m['train_s']:>7.0f}s", flush=True)
        print(f"     read: {m['sample'][0]!r}\n     true: {m['sample'][1]!r}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 4, 3, 2])
    a = ap.parse_args()
    main(a.steps, a.sigma, a.sizes)
