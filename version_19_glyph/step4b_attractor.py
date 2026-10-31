"""step4b_attractor.py — an attractor canvas: store, read and complete in one settle.

The canvas is an image whose rows are glyph patches (glyph_net, P=4, 64 chars
in 48 pixels). Storing a memory = writing its patch into the canvas. Reading
= writing a *partial* cue (half a passage, or a passage with typos) and
letting it settle against every stored patch at once:

    q <- C^T softmax(beta * cos(C, q))        (modern Hopfield = attention)

then the reader decodes the settled patch into the full 64 characters. No
index, no search loop: the cue falls into the basin of its memory in one
parallel step over the whole canvas.

The writer is trained so that it can be read back (reconstruction) *and* so
that partial cues land near their full memory (an InfoNCE term), which is
what carves the basins.

Run:  .venv/bin/python step4b_attractor.py [--steps 4000]
"""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import glyph_net as gn
from step2_codec import TEST, TRAIN, book
from step4a_parallel import L, windows

P = 4
MASK = gn.V - 1


def cue(x, rng, kind=None):
    """Partial view of x: a prefix (rest masked) or ~15% of chars corrupted."""
    x = x.copy()
    kinds = rng.integers(0, 2, len(x)) if kind is None else np.full(len(x), kind)
    for i, k in enumerate(kinds):
        if k == 0:
            x[i, rng.integers(L // 4, 3 * L // 4):] = MASK
        else:
            hit = rng.random(L) < 0.15
            x[i, hit] = rng.integers(0, gn.V - 1, hit.sum())
    return x


def flat(img):
    v = img.reshape(img.shape[0], -1) - 0.5
    return v / (mx.linalg.norm(v, axis=1, keepdims=True) + 1e-6)


class Canvas(nn.Module):
    def __init__(self):
        super().__init__()
        self.w, self.r = gn.Writer(L, P), gn.Reader(L, P)
        self.log_beta = mx.array(np.log(20.0))


def settle(canvas, q, beta, iters=1):
    """canvas: (N, P, P, 3) stored patches, q: (B, P, P, 3) cue patches."""
    C, Cf = canvas.reshape(canvas.shape[0], -1), flat(canvas)
    for _ in range(iters):
        a = mx.softmax(beta * flat(q) @ Cf.T, axis=1)
        q = (a @ C).reshape(q.shape)
    return q, a


def train(steps, train_ids, batch=256):
    mx.random.seed(0)
    rng = np.random.default_rng(0)
    m = Canvas()
    warm = min(200, steps // 4)
    opt = optim.AdamW(optim.join_schedules(
        [optim.linear_schedule(1e-5, 5e-4, warm),
         optim.cosine_decay(5e-4, steps - warm, 1e-5)], [warm]), weight_decay=0.01)

    def loss_fn(m, x, c):
        full = gn.channel(m.w(x), 0, True)
        rec = nn.losses.cross_entropy(m.r(full), x).mean()
        sim = mx.exp(m.log_beta) * flat(gn.channel(m.w(c), 0, True)) @ flat(full).T
        nce = nn.losses.cross_entropy(sim, mx.arange(len(x))).mean()
        return rec + nce

    step = nn.value_and_grad(m, loss_fn)
    for _ in range(steps):
        xb = windows(train_ids, batch, rng)
        loss, g = step(m, mx.array(xb), mx.array(cue(xb, rng)))
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)
    return m, float(loss)


def evaluate(m, bank_ids, n_store, kind, n_query=1000, beta=None, iters=1):
    rng = np.random.default_rng(1)
    starts = np.arange(0, len(bank_ids) - L, L)[:n_store]
    stored = bank_ids[starts[:, None] + np.arange(L)]
    canvas = mx.concatenate([gn.channel(m.w(mx.array(stored[i:i + 1024])), 0, False)
                             for i in range(0, len(stored), 1024)])
    qi = rng.choice(len(stored), n_query, replace=False)
    c = cue(stored[qi], rng, kind)
    t = time.time()
    q = gn.channel(m.w(mx.array(c)), 0, False)
    beta = beta or float(mx.exp(m.log_beta)) * 4
    out, a = settle(canvas, q, beta, iters)
    pred = np.array(mx.argmax(m.r(out), -1))
    dt = time.time() - t
    hit = (np.array(mx.argmax(a, 1)) == qi).mean()
    exact = (pred == stored[qi]).all(1).mean()
    return hit, exact, n_query / dt, gn.text(c[0]), gn.text(pred[0])


def main(steps):
    train_ids = gn.ids("\n\n".join(book(k, v) for k, v in TRAIN.items()))
    test_ids = gn.ids(book("pride_and_prejudice", TEST["pride_and_prejudice"]))
    t = time.time()
    m, loss = train(steps, train_ids)
    print(f"trained {steps} steps in {time.time() - t:.0f}s, loss {loss:.3f}, "
          f"beta {float(mx.exp(m.log_beta)):.1f}\n")

    # memories are held-out Pride and Prejudice; past ~11k, train text joins as distractors
    bank = np.concatenate([test_ids, train_ids])
    print(f"{'memories':>9}{'canvas px':>11}{'cue':>14}{'basin hit':>11}"
          f"{'exact recall':>14}{'queries/s':>11}")
    for n in (1_000, 10_000, 50_000):
        for kind, name in ((0, "prefix 25-75%"), (1, "15% typos")):
            hit, ex, qps, c, p = evaluate(m, bank, n, kind)
            print(f"{n:>9,}{n * P * P:>11,}{name:>14}{hit:>11.3f}{ex:>14.3f}{qps:>11,.0f}",
                  flush=True)
    print(f"\ncue:     {c!r}\nsettled: {p!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--P", type=int, default=P)
    a = ap.parse_args()
    P = a.P
    main(a.steps)
