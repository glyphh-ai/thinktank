"""lottery.py — can a think-tank language model predict Texas Lottery numbers?

Rules, split, models and pass bar are fixed in PROTOCOL.md (written before
this ran). Every game becomes text, one byte per number, one line per draw,
and four predictors score the test draws: uniform (a fair draw), hot
numbers, bitspin (version_21, learns while it reads) and a small byte
transformer (version_21 ByteLM, backprop, frozen on test).

Run:  ../version_21_bitspin/.venv/bin/python lottery.py [--quick]
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "version_21_bitspin"))
import bitspin  # noqa: E402

DATA = os.path.join(HERE, "data")
SLOTS = {"Morning": 0, "Day": 1, "Evening": 2, "Night": 3}
N_TESTS = 15                       # 5 games x 3 models (Bonferroni)
ALPHA = 0.01 / N_TESTS


# ---------------------------------------------------------------- data

class Game:
    """kind 'digits': n digits 0-9.  kind 'balls': n of pool, draw order,
    plus an optional bonus ball from its own pool."""

    def __init__(self, name, kind, n, pool=10, bonus=0, draws=()):
        self.name, self.kind, self.n, self.pool, self.bonus = name, kind, n, pool, bonus
        self.draws = [list(d) for d in draws]

    def width(self):
        return self.n + (1 if self.bonus else 0)

    def byte(self, v):
        return ord("0") + v if self.kind == "digits" else 64 + v

    def text(self, draws=None):
        return b"".join(bytes(self.byte(v) for v in d) + b"\n"
                        for d in (self.draws if draws is None else draws))

    def legal(self, j, prefix):
        """Values the j-th number can take, given the draw's numbers before it."""
        if self.kind == "digits":
            return list(range(10))
        if j < self.n:
            return [v for v in range(1, self.pool + 1) if v not in prefix[:j]]
        return list(range(1, self.bonus + 1))

    def uniform_bits(self):
        if self.kind == "digits":
            return self.n * math.log2(10)
        b = sum(math.log2(self.pool - k) for k in range(self.n))
        return b + (math.log2(self.bonus) if self.bonus else 0)


def read(files, width, since=(0, 0, 0)):
    rows = []
    for f in files:
        for r in csv.reader(open(os.path.join(DATA, f))):
            m, d, y = int(r[1]), int(r[2]), int(r[3])
            if (y, m, d) < since:
                continue
            slot = next((v for k, v in SLOTS.items() if r[0].endswith(k)), 0)
            rows.append(((y, m, d, slot), [int(x) for x in r[4:4 + width]]))
    rows.sort()
    return [v for _, v in rows]


def games():
    p3 = [f"pick3{s}.csv" for s in ("morning", "day", "evening", "night")]
    d4 = [f"daily4{s}.csv" for s in ("morning", "day", "evening", "night")]
    return [
        Game("Lotto Texas", "balls", 6, 54, draws=read(["lottotexas.csv"], 6, (2006, 5, 1))),
        Game("Cash Five", "balls", 5, 35, draws=read(["cashfive.csv"], 5, (2018, 9, 24))),
        Game("Texas Two Step", "balls", 4, 35, bonus=35, draws=read(["texastwostep.csv"], 5)),
        Game("Pick 3", "digits", 3, draws=read(p3, 3)),
        Game("Daily 4", "digits", 4, draws=read(d4, 4)),
    ]


def controls(n_draws):
    rng = np.random.default_rng(23)
    neg = rng.integers(0, 10, (n_draws, 3)).tolist()
    rng = np.random.default_rng(24)
    pos = rng.integers(0, 10, (n_draws, 3))
    copy = rng.random(n_draws) < 0.30
    for i in range(1, n_draws):
        if copy[i]:
            pos[i, 0] = pos[i - 1, 2]
    return [Game("control: random", "digits", 3, draws=neg),
            Game("control: planted", "digits", 3, draws=pos.tolist())]


def split(game):
    n = len(game.draws)
    a, b = int(n * 0.7), int(n * 0.8)
    return game.draws[:a], game.draws[a:b], game.draws[b:]


# ---------------------------------------------------------------- models
# Each model's run(game, train, valid, test) returns, per test draw,
# (bits of the actual draw, the ticket it would have bought before the draw).

def run_uniform(game, train, valid, test):
    rng = np.random.default_rng(0)
    out = []
    for _ in test:
        t = []
        for j in range(game.width()):
            t.append(int(rng.choice(game.legal(j, t))))
        out.append((game.uniform_bits(), t))
    return out


def fresh_counts(game):
    """Count table per number slot j. Digits: one per position. Balls: every
    main ball shares one table ("17 is hot", any position), the bonus has
    its own."""
    if game.kind == "digits":
        return [np.zeros(10) for _ in range(game.n)]
    main = np.zeros(game.pool + 1)
    return [main] * game.n + ([np.zeros(game.bonus + 1)] if game.bonus else [])


def hot_pass(game, draws, a, d, counts):
    """Online: predict each draw from decayed counts, then count it."""
    tables = list({id(c): c for c in counts}.values())
    out = []
    for dr in draws:
        bits, ticket = 0.0, []
        for j, v in enumerate(dr):
            c = counts[j]
            leg = game.legal(j, dr)
            w = c[leg] + a
            bits -= math.log2(w[leg.index(v)] / w.sum())
            legt = game.legal(j, ticket)
            ticket.append(legt[int(np.argmax(c[legt]))])
        for c in tables:
            c *= d
        for j, v in enumerate(dr):
            counts[j][v] += 1
        out.append((bits, ticket))
    return out


def run_hot(game, train, valid, test):
    best = None
    for a in (1.0, 10.0, 100.0, 1000.0):
        for d in (1.0, 0.999, 0.99, 0.95):
            c = fresh_counts(game)
            hot_pass(game, train, a, d, c)
            v = sum(b for b, _ in hot_pass(game, valid, a, d, c))
            if best is None or v < best[0]:
                best = (v, a, d)
    _, a, d = best
    c = fresh_counts(game)
    hot_pass(game, train + valid, a, d, c)
    out = hot_pass(game, test, a, d, c)
    run_hot.settings = {"prior": a, "decay": d}
    return out


def bitspin_probe(m, hist_bytes, cands):
    """-log2 p of each candidate byte right after hist_bytes, counters as
    they stand. The same arithmetic as BitSpin.run's prediction (checked
    against it on every scored byte), for many candidate bytes at once."""
    pad = m.maxlag
    ctx = hist_bytes[-4096:].rjust(pad, b"\0")
    hist = np.frombuffer(ctx + b"\0", np.uint8).astype(np.int64)
    words = bitspin.word_state(ctx + b"\0", m.maxwords) if m.maxwords else None
    t = np.full(len(cands), len(ctx))
    c = np.array(cands, np.int64)
    i = np.arange(8)
    y = ((c[:, None] >> (7 - i)) & 1).astype(np.int64)
    prefix = (1 << i) | (c[:, None] >> (8 - i))
    prefix[:, 0] = 1
    ps, cls, words_n = [], [], []
    chain = np.full(y.shape, 0.5)
    depth = np.zeros(y.shape, np.int64)
    for j, h in enumerate(m.heads):
        idx, tag = h.slots(h.keys(m.hv, hist, t, words), prefix)
        n = h.read(idx, tag)
        tot = n[..., 0] + n[..., 1]
        if h.kind == "exact":
            beta = m.beta * ((n[..., 0] > 0).astype(float) + (n[..., 1] > 0))
            seen = tot > 0
            chain = np.where(seen, (n[..., 1] + beta * chain) / (tot + beta + 1e-12), chain)
            depth = np.where(seen, j + 1, depth)
        elif h.kind == "memory":
            words_n.append(n)
        else:
            ps.append((n[..., 1] + m.prior) / (tot + 2 * m.prior))
            cls.append(bitspin.state_class(n))
    ps.insert(0, chain)
    cls.insert(0, np.minimum(depth, m.levels - 1))
    P, C = np.stack(ps), np.stack(cls)
    hidx = np.arange(len(P))[:, None, None]
    W = np.exp(m.logw[hidx, C])
    p1 = np.clip((W * P).sum(0) / W.sum(0), 1e-6, 1 - 1e-6)
    for n in words_n:
        tot = n[..., 0] + n[..., 1]
        p1 = np.where(tot > 0, (n[..., 1] + m.beta_word * p1) / (tot + m.beta_word), p1)
    p1 = np.clip(p1, 1e-6, 1 - 1e-6)
    return -np.log2(np.where(y == 1, p1, 1 - p1)).sum(1)


def run_bitspin(game, train, valid, test):
    m = bitspin.BitSpin()
    hist = game.text(train + valid)
    m.run(hist, learn=True, batch=256)
    out, worst = [], 0.0
    for dr in test:
        ticket = []
        for j in range(game.width()):
            leg = game.legal(j, ticket)
            pre = bytes(game.byte(v) for v in ticket)
            b = bitspin_probe(m, hist + pre, [game.byte(v) for v in leg])
            ticket.append(leg[int(np.argmin(b))])
        line = game.text([dr])
        bits = sum(float(bitspin_probe(m, hist + line[:j], [line[j]])[0])
                   for j in range(len(dr)))
        m.run(line, learn=True, batch=len(line), tail=hist)
        worst = max(worst, abs(m.last_per_byte[:len(dr)].sum() - bits))
        hist += line
        out.append((bits, ticket))
    assert worst < 1e-6, f"probe disagrees with BitSpin.run by {worst}"
    return out


CTX = 256


class ByteLM(nn.Module):
    """version_21 step2 ByteLM, smaller."""

    def __init__(self, d=128, layers=4, heads=4, dropout=0.1):
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


def tf_windows(full, ends):
    """Windows of CTX bytes ending right before each position in `ends`."""
    x = np.frombuffer(full, np.uint8).astype(np.int32)
    return np.stack([x[e - CTX:e] if e >= CTX else np.pad(x[:e], (CTX - e, 0), constant_values=10)
                     for e in ends])


def tf_logp(model, windows):
    """log p of the byte after each window."""
    out = []
    for k in range(0, len(windows), 256):
        lp = nn.log_softmax(model(mx.array(windows[k:k + 256]))[:, -1, :], axis=-1)
        out.append(np.array(lp))
    return np.concatenate(out)


def run_transformer(game, train, valid, test, max_steps=6000, quick=False):
    mx.random.seed(0)
    rng = np.random.default_rng(0)
    model = ByteLM()
    tr, va = game.text(train), game.text(valid)
    x = np.frombuffer(tr, np.uint8).astype(np.int32)
    xv = np.frombuffer(tr[-CTX:] + va, np.uint8).astype(np.int32)
    vwin = np.stack([xv[a:a + CTX + 1] for a in range(0, len(xv) - CTX - 1, 64)])
    opt = optim.AdamW(1e-3, weight_decay=0.1)

    def loss_fn(mdl, xb, yb):
        return nn.losses.cross_entropy(mdl(xb), yb, reduction="mean")

    step = nn.value_and_grad(model, loss_fn)
    best, best_w, bad = 1e9, None, 0
    for i in range(300 if quick else max_steps):
        a = rng.integers(0, len(x) - CTX - 1, 32)
        win = np.stack([x[j:j + CTX + 1] for j in a])
        _, g = step(model, mx.array(win[:, :-1]), mx.array(win[:, 1:]))
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state)
        if (i + 1) % 200 == 0:
            model.eval()
            v = sum(loss_fn(model, mx.array(vwin[k:k + 64, :-1]), mx.array(vwin[k:k + 64, 1:])).item()
                    for k in range(0, len(vwin), 64))
            model.train()
            if v < best:
                best, best_w, bad = v, dict(nn.utils.tree_flatten(model.parameters())), 0
            else:
                bad += 1
                if bad >= 5:
                    break
    model.update(nn.utils.tree_unflatten(list(best_w.items())))
    model.eval()
    run_transformer.steps = i + 1
    hist = game.text(train + valid)
    full = hist + game.text(test)
    starts, p = [], len(hist)
    for dr in test:
        starts.append(p)
        p += len(dr) + 1
    starts = np.array(starts)
    bits = np.zeros(len(test))
    tickets = [[] for _ in test]
    for j in range(game.width()):
        win = tf_windows(full, starts + j)                 # real history, real prefix
        lp = tf_logp(model, win)
        for k, dr in enumerate(test):
            leg = [game.byte(v) for v in game.legal(j, dr)]
            z = lp[k, leg]
            bits[k] -= (z[leg.index(game.byte(dr[j]))] - np.logaddexp.reduce(z)) / math.log(2)
        # ticket: real history, the ticket's own numbers so far as the prefix
        if j:
            win[:, CTX - j:] = [[game.byte(v) for v in t] for t in tickets]
        lp = tf_logp(model, win)
        for k in range(len(test)):
            leg = game.legal(j, tickets[k])
            tickets[k].append(leg[int(np.argmax(lp[k, [game.byte(v) for v in leg]]))])
    return list(zip(bits.tolist(), tickets))


# ---------------------------------------------------------------- scoring

def p_upper(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def binom_upper(k, n, p):
    """P(X >= k), X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    lt = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
          + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k, n + 1)]
    mx_ = max(lt)
    return min(1.0, math.exp(mx_) * sum(math.exp(v - mx_) for v in lt))


def money(game, test, tickets):
    if game.kind == "digits":
        hits = sum(t == d for t, d in zip(tickets, test))
        n, p = len(test), 10.0 ** -game.n
        return {"hits": hits, "expected": n * p, "p": binom_upper(hits, n, p)}
    k, N = game.n, game.pool
    hits = sum(len(set(t[:k]) & set(d[:k])) for t, d in zip(tickets, test))
    mean = k * k / N
    var = k * (k / N) * ((N - k) / N) * ((N - k) / (N - 1))
    n = len(test)
    z = (hits - n * mean) / math.sqrt(n * var)
    return {"hits": hits, "expected": n * mean, "p": p_upper(z)}


def judge(game, test, out):
    u = game.uniform_bits()
    g = np.array([u - b for b, _ in out])
    n = len(g)
    se = g.std(ddof=1) / math.sqrt(n) if g.std() > 0 else 1e-12
    h = n // 2
    mon = money(game, test, [t for _, t in out])
    r = {"draws": n, "gain_bits_per_draw": float(g.mean()), "p": p_upper(g.mean() / se),
         "gain_first_half": float(g[:h].mean()), "gain_second_half": float(g[h:].mean()),
         "money": mon}
    r["pass"] = bool(r["p"] < ALPHA and r["gain_first_half"] > 0
                     and r["gain_second_half"] > 0 and mon["p"] < ALPHA)
    return r


def main(quick):
    gs = games()
    gs += controls(len(gs[3].draws))
    models = [("uniform", run_uniform), ("hot numbers", run_hot),
              ("bitspin", run_bitspin),
              ("transformer", lambda *a: run_transformer(*a, quick=quick))]
    results = {}
    for g in gs:
        tr, va, te = split(g)
        if quick:
            te = te[:200]
        print(f"\n== {g.name}: {len(g.draws):,} draws (train {len(tr):,} / valid {len(va):,} / "
              f"test {len(te):,}), uniform {g.uniform_bits():.3f} bits per draw", flush=True)
        results[g.name] = {}
        for name, fn in models:
            t0 = time.time()
            r = judge(g, te, fn(g, tr, va, te))
            if name == "hot numbers":
                r["settings"] = run_hot.settings
            if name == "transformer":
                r["train_steps"] = run_transformer.steps
            results[g.name][name] = r
            m = r["money"]
            print(f"  {name:<12} gain {r['gain_bits_per_draw']:+.4f} bits/draw  p {r['p']:.3g}  "
                  f"halves {r['gain_first_half']:+.4f} {r['gain_second_half']:+.4f}  "
                  f"ticket {m['hits']} vs {m['expected']:.1f} (p {m['p']:.3g})  "
                  f"{'PASS' if r['pass'] else 'fail'}  {time.time() - t0:.0f}s", flush=True)
    if not quick:
        json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test: 200 test draws, 300 steps")
    main(ap.parse_args().quick)
