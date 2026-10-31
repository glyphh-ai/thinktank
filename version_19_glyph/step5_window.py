"""step5_window.py — does a model that reads the canvas directly explode the window?

Each episode is N invented facts ("Bolani lives in Tesaruk."), one 64-char
fact per glyph patch. A question model cross-attends to *every* patch of the
canvas while it answers, so nothing is retrieved first. Two kinds of question:

  recall:    "Where does Bolani live?"        one memory holds the answer
  aggregate: "How many live in Tesaruk?"      the answer is spread over many
             memories, so top-1 retrieval (step4b) can't answer it at all

Names and cities are random every episode, so nothing can be memorized.
Trained on 4 <= N <= 128 facts, tested up to N = 4096 (262k chars).

Two arms, same question model:
  canvas : memory = one token per fact, read from its 8×8 RGB patch (8-bit)
  window : memory = the raw characters, a 4,096-char context window (64 facts);
           past that it sees only the last 64 facts, like a real window

Run:  .venv/bin/python step5_window.py [--steps 3000]
"""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import glyph_net as gn

L, P, QL, AL = 64, 8, 48, 16          # fact chars, patch side, question, answer
WINDOW_FACTS = 64                     # 4,096 chars
D = 192
CV = "bdfgklmnprstvz", "aeiou"


def word(rng, syl):
    return "".join(rng.choice(list(CV[0])) + rng.choice(list(CV[1]))
                   for _ in range(syl)).capitalize()


def episode(rng, n):
    names, cities = set(), set()
    while len(names) < n:
        names.add(word(rng, int(rng.integers(2, 5))))
    nc = max(2, n // 8)                         # ~8 people per city at any N
    while len(cities) < nc:
        cities.add(word(rng, 3) + rng.choice(["k", "n", "r"]))
    names, cities = sorted(names), sorted(cities)
    rng.shuffle(names)
    home = {nm: cities[int(rng.integers(nc))] for nm in names}
    facts = [f"{nm} lives in {home[nm]}." for nm in names]
    if rng.random() < 0.5:
        nm = names[int(rng.integers(n))]
        q, a, kind = f"Where does {nm} live?", home[nm], 0
    else:
        c = home[names[int(rng.integers(n))]]
        q, a, kind = f"How many live in {c}?", str(sum(v == c for v in home.values())), 1
    return facts, q, a, kind


def pad(s, n):
    return gn.ids(s.ljust(n)[:n])


def batch(rng, b, n):
    eps = [episode(rng, n) for _ in range(b)]
    F = np.stack([np.stack([pad(f, L) for f in e[0]]) for e in eps])   # (B, N, L)
    Q = np.stack([pad(e[1], QL) for e in eps])
    A = np.stack([pad(e[2], AL) for e in eps])
    K = np.array([e[3] for e in eps])
    return F, Q, A, K


class XBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)
        self.self_attn = nn.MultiHeadAttention(D, 6)
        self.cross = nn.MultiHeadAttention(D, 6)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def __call__(self, q, m):
        h = self.n1(q)
        q = q + self.self_attn(h, h, h)
        q = q + self.cross(self.n2(q), m, m)
        return q + self.mlp(self.n3(q))


class Asker(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.emb = nn.Embedding(gn.V, D)
        self.qpos = mx.random.normal((QL + AL, D)) * 0.02
        if arm == "canvas":
            self.writer = gn.Writer(L, P, d=128, layers=2, heads=4)
            self.read = nn.Sequential(nn.Linear(P * P * 3, D), nn.GELU(), nn.Linear(D, D))
        else:
            self.fpos = mx.random.normal((L, D)) * 0.02
        self.blocks = [XBlock() for _ in range(4)]
        self.out = nn.Linear(D, gn.V)

    def memory(self, F, train):
        B, N, _ = F.shape
        if self.arm == "canvas":
            img = gn.channel(self.writer(F.reshape(B * N, L)), 0, train)
            return self.read(img.reshape(B, N, -1))              # (B, N, D)
        F = F[:, -WINDOW_FACTS:]
        return (self.emb(F) + self.fpos).reshape(B, -1, D)       # (B, <=4096, D)

    def __call__(self, F, Q, train=False):
        m = self.memory(F, train)
        slots = mx.full((Q.shape[0], AL), gn.V - 1, mx.int32)
        q = self.emb(mx.concatenate([Q, slots], 1)) + self.qpos
        for blk in self.blocks:
            q = blk(q, m)
        return self.out(q[:, QL:])


def train(arm, steps, b=32):
    mx.random.seed(0)
    rng = np.random.default_rng(0)
    m = Asker(arm)
    warm = min(200, steps // 4)
    opt = optim.AdamW(optim.join_schedules(
        [optim.linear_schedule(1e-5, 6e-4, warm),
         optim.cosine_decay(6e-4, steps - warm, 1e-5)], [warm]), weight_decay=0.01)

    def loss_fn(m, F, Q, A):
        return nn.losses.cross_entropy(m(F, Q, True), A).mean()

    step = nn.value_and_grad(m, loss_fn)
    for i in range(steps):
        n = int(rng.choice([4, 8, 16, 32, 64, 128]))   # small N first teaches lookup
        F, Q, A, _ = batch(rng, b, n)
        loss, g = step(m, mx.array(F), mx.array(Q), mx.array(A))
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)
    return m, float(loss)


def evaluate(m, n, episodes=200):
    rng = np.random.default_rng(10_000 + n)
    hits, kinds = [], []
    per = max(1, 4096 // n)
    for i in range(0, episodes, per):
        F, Q, A, K = batch(rng, min(per, episodes - i), n)
        pred = np.array(mx.argmax(m(mx.array(F), mx.array(Q)), -1))
        hits += list((pred == A).all(1))
        kinds += list(K)
    hits, kinds = np.array(hits), np.array(kinds)
    return hits[kinds == 0].mean(), hits[kinds == 1].mean()


def main(steps):
    print("train N in {4..128}; test held-out episodes, exact answer\n")
    print(f"{'arm':<8}{'N facts':>9}{'chars':>10}{'mem tokens':>12}"
          f"{'recall':>9}{'aggregate':>11}")
    for arm in ("canvas", "window"):
        t = time.time()
        m, loss = train(arm, steps)
        for n in (16, 64, 256, 1024, 4096):
            r, a = evaluate(m, n)
            toks = n if arm == "canvas" else min(n, WINDOW_FACTS) * L
            print(f"{arm:<8}{n:>9,}{n * L:>10,}{toks:>12,}{r:>9.3f}{a:>11.3f}", flush=True)
        print(f"{'':<8}trained {steps} steps in {time.time() - t:.0f}s, loss {loss:.3f}\n",
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    main(ap.parse_args().steps)
