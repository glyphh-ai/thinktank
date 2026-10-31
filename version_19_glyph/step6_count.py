"""step6_count.py — aggregation inside the settle: can the canvas count?

step4b showed one settle finds a memory from a partial cue. step5 tried to
make a learned question model aggregate and couldn't train. Here aggregation
is read straight off the attractor, with no question model at all:

  "Where does Bolani live?"   cue "Bolani lives in ________." -> settle -> read
  "Who lives in Tesaruk?"     cue "________ lives in Tesaruk." -> settle -> read
  "How many live in Tesaruk?" same cue; count = how many memories the cue's
                              similarity puts above a threshold calibrated once
                              on separate episodes

The writer is retrained (P=8, 1 char/px) on prose plus invented facts, with
cues that blank *any* span (start, middle or end) or scramble 15% of chars,
so the masked-name cue lands in the basin of every fact sharing the city.

Run:  .venv/bin/python step6_count.py [--steps 12000]
"""

import argparse
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import glyph_net as gn
import step4b_attractor as att
from step2_codec import TRAIN, book
from step4a_parallel import L, windows
from step5_window import pad, word

att.P = 8
MASK = gn.V - 1


NAME_W = 10                            # fixed-width name slot, so residents align


def fact(name, city):
    return f"{name:<{NAME_W}}lives in {city}."


def facts_batch(rng, b):
    out = []
    while len(out) < b:
        names, cities = set(), set()
        while len(names) < 16:
            names.add(word(rng, int(rng.integers(2, 5))))
        while len(cities) < 2:
            cities.add(word(rng, 3) + rng.choice(["k", "n", "r"]))
        cities = sorted(cities)
        out += [fact(nm, cities[int(rng.integers(2))]) for nm in sorted(names)]
    return np.stack([pad(f, L) for f in out[:b]])


def cue_any(x, rng, n_facts=0):
    """Blank a span anywhere or scramble 15% of chars. The last n_facts rows
    are facts and also get slot cues: name blanked, or city blanked."""
    x = x.copy()
    first_fact = len(x) - n_facts
    for i in range(len(x)):
        u = rng.random()
        if i >= first_fact and u < 0.3:
            x[i, :NAME_W] = MASK
            continue
        if i >= first_fact and u < 0.45:
            x[i, NAME_W + 9:] = MASK
            continue
        if u > 0.8:
            hit = rng.random(L) < 0.15
            x[i, hit] = rng.integers(0, gn.V - 1, hit.sum())
            continue
        n = int((x[i] != gn.ids(" ")[0]).nonzero()[0].max()) + 1   # text length
        span = int(rng.integers(max(1, n // 4), max(2, 3 * n // 4)))
        s = int(rng.integers(0, max(1, n - span)))
        x[i, s:s + span] = MASK
    return x


def positives(x, c):
    """pos[i, j] = memory j agrees with cue i on every unblanked char. A
    scrambled cue agrees with nothing, so it falls back to its own memory."""
    vis = c != MASK
    pos = ((x[None, :, :] == c[:, None, :]) | ~vis[:, None, :]).all(2)
    pos[np.arange(len(x)), np.arange(len(x))] = True
    return pos


def train(steps, prose, batch=256):
    mx.random.seed(0)
    rng = np.random.default_rng(0)
    m = att.Canvas()
    warm = min(200, steps // 4)
    opt = optim.AdamW(optim.join_schedules(
        [optim.linear_schedule(1e-5, 5e-4, warm),
         optim.cosine_decay(5e-4, steps - warm, 1e-5)], [warm]), weight_decay=0.01)

    def loss_fn(m, x, c, pos):
        full = gn.channel(m.w(x), 0, True)
        rec = nn.losses.cross_entropy(m.r(full), x).mean()
        sim = mx.exp(m.log_beta) * att.flat(gn.channel(m.w(c), 0, True)) @ att.flat(full).T
        # every consistent memory is a positive, each pulled in equally
        logp = sim - mx.logsumexp(sim, axis=1, keepdims=True)
        nce = -(mx.where(pos, logp, 0).sum(1) / pos.sum(1)).mean()
        return rec + nce

    step = nn.value_and_grad(m, loss_fn)
    for _ in range(steps):
        x = np.concatenate([windows(prose, batch // 2, rng), facts_batch(rng, batch // 2)])
        c = cue_any(x, rng, batch // 2)
        loss, g = step(m, mx.array(x), mx.array(c), mx.array(positives(x, c)))
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)
    return m, float(loss)


def write(m, rows):
    return mx.concatenate([gn.channel(m.w(mx.array(rows[i:i + 1024])), 0, False)
                           for i in range(0, len(rows), 1024)])


def probe(m, rng, n):
    """One episode of n facts: recall by name, recall by city, and count by city."""
    names, cities = set(), set()
    while len(names) < n:
        names.add(word(rng, int(rng.integers(2, 5))))
    nc = max(2, n // 8)
    while len(cities) < nc:
        cities.add(word(rng, 3) + rng.choice(["k", "n", "r"]))
    names, cities = sorted(names), sorted(cities)
    home = {nm: cities[int(rng.integers(nc))] for nm in names}
    facts = [fact(nm, home[nm]) for nm in names]
    rows = np.stack([pad(f, L) for f in facts])
    canvas = write(m, rows)
    Cf = att.flat(canvas)

    q = 64
    who = rng.choice(n, q, replace=False)
    by_name, by_city = rows[who].copy(), rows[who].copy()
    by_name[:, NAME_W + 9:] = MASK                     # "Bolani    lives in ____"
    by_city[:, :NAME_W] = MASK                         # "__________lives in Tesaruk."
    true_count = np.array([sum(v == home[names[i]] for v in home.values()) for i in who])

    beta = float(mx.exp(m.log_beta)) * 4
    out, _ = att.settle(canvas, write(m, by_name), beta)
    pred = np.array(mx.argmax(m.r(out), -1))
    name_exact = np.mean([gn.text(pred[k]).strip() == facts[i] for k, i in enumerate(who)])

    qc = write(m, by_city)
    out, _ = att.settle(canvas, qc, beta)
    pred = np.array(mx.argmax(m.r(out), -1))
    city_ok = np.mean([gn.text(pred[k]).strip().endswith(f"lives in {home[names[i]]}.")
                       for k, i in enumerate(who)])
    sims = np.array(att.flat(qc) @ Cf.T)                          # (q, n)
    same = np.array([[home[nm] == home[names[i]] for nm in names] for i in who])
    # separable = every resident scores above every non-resident for this cue
    sep = np.mean([s[t].min() > s[~t].max() for s, t in zip(sims, same)])
    return name_exact, city_ok, sims, true_count, sep


def main(steps):
    prose = gn.ids("\n\n".join(book(k, v) for k, v in TRAIN.items()))
    t = time.time()
    m, loss = train(steps, prose)
    print(f"trained {steps} steps in {time.time() - t:.0f}s, loss {loss:.3f}\n")
    m.save_weights(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "step6_canvas.safetensors"))

    # calibrate the count threshold once, on separate episodes
    rng = np.random.default_rng(123)
    cal = [probe(m, rng, 256) for _ in range(4)]
    grid = np.linspace(-0.2, 1.0, 241)
    acc = [np.mean(np.concatenate([(r[2] > g).sum(1) == r[3] for r in cal])) for g in grid]
    tau = float(grid[int(np.argmax(acc))])
    print(f"count threshold tau = {tau:.3f} (calibrated on 4 episodes of 256 facts)\n")

    print(f"{'facts':>7}{'name->fact exact':>18}{'city->a resident':>18}"
          f"{'count exact':>13}{'count +-1':>11}{'separable':>11}{'mean count':>12}")
    rng = np.random.default_rng(7)
    for n in (64, 256, 1024, 4096, 16384):
        res = [probe(m, rng, n) for _ in range(3)]
        ne = np.mean([r[0] for r in res])
        co = np.mean([r[1] for r in res])
        est = np.concatenate([(r[2] > tau).sum(1) for r in res])
        tru = np.concatenate([r[3] for r in res])
        print(f"{n:>7,}{ne:>18.3f}{co:>18.3f}{np.mean(est == tru):>13.3f}"
              f"{np.mean(abs(est - tru) <= 1):>11.3f}{np.mean([r[4] for r in res]):>11.3f}"
              f"{tru.mean():>12.1f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    main(ap.parse_args().steps)
