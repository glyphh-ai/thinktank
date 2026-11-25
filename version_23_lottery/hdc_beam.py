"""hdc_beam.py — Addendum A of PROTOCOL.md: an HDC model, tickets by beam search.

Each number is predicted from a hypervector of the last K draws (bipolar,
version_06 hdc.py, D = 10,000): bind(role(lag, position), value) for every
number, majority-bundled, plus the numbers already drawn in this draw. A
prototype per (slot, value) sums the contexts that came before that value;
the prediction is cosine to each prototype, standardized, p ∝ exp(β z).
K and β come from validation. Tickets: beam search, width 16.

Run:  ../version_21_bitspin/.venv/bin/python hdc_beam.py [--quick]
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "version_06_hdc_forward"))
import hdc  # noqa: E402
import lottery  # noqa: E402

lottery.ALPHA = 0.01 / 20
KS = (1, 2, 4)
BETAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
BEAM = 16
TOP = 10


class HDC:
    def __init__(self, game, K):
        self.g, self.K = game, K
        enc = hdc.Encoder()
        w, nv = game.width(), max(game.pool, game.bonus) + 1
        # bound[k, i, v] = bind(role(lag k, position i), value v); lag 0 = this draw
        self.bound = np.stack([np.stack([np.stack([
            hdc.bind(enc.role_vec(f"lag{k}_pos{i}"), enc.value_vec(str(v))) for v in range(nv)])
            for i in range(w)]) for k in range(K + 1)]).astype(np.int32)
        nslots = w if game.kind == "digits" else (2 if game.bonus else 1)
        self.mem = np.zeros((nslots, nv, enc.D))

    def slot(self, j):
        return j if self.g.kind == "digits" else (0 if j < self.g.n else 1)

    def history(self, draws, t):
        s = np.zeros(self.bound.shape[-1], np.int32)
        for k in range(1, self.K + 1):
            for i, v in enumerate(draws[t - k]):
                s += self.bound[k, i, v]
        return s

    def cortex(self, hist, prefix):
        s = hist.copy()
        for i, v in enumerate(prefix):
            s += self.bound[0, i, v]
        return np.where(s >= 0, 1.0, -1.0)

    def z(self, j, cx, legal):
        P = self.mem[self.slot(j), legal]
        P = P - P.mean(0)
        s = P @ cx / (np.linalg.norm(P, axis=1) + 1e-9)
        sd = s.std()
        return (s - s.mean()) / sd if sd > 0 else np.zeros(len(legal))

    def learn(self, draws, t, hist):
        for j, v in enumerate(draws[t]):
            self.mem[self.slot(j), v] += self.cortex(hist, draws[t][:j])


def logp(beta, z):
    a = beta * z
    return a - np.logaddexp.reduce(a)


def online(m, draws, lo, hi, keep=False):
    """Learn draws[lo:hi] in order; with keep, first record (z, index of the
    actual value) for every number."""
    rec = []
    for t in range(max(lo, m.K), hi):
        hist = m.history(draws, t)
        if keep:
            for j, v in enumerate(draws[t]):
                leg = m.g.legal(j, draws[t])
                rec.append((m.z(j, m.cortex(hist, draws[t][:j]), leg), leg.index(v)))
        m.learn(draws, t, hist)
    return rec


def beam(m, hist):
    beams = [([], 0.0)]
    for j in range(m.g.width()):
        nxt = []
        for tk, sc in beams:
            leg = m.g.legal(j, tk)
            lp = logp(m.beta, m.z(j, m.cortex(hist, tk), leg))
            nxt += [(tk + [v], sc + float(lp[i])) for i, v in enumerate(leg)]
        nxt.sort(key=lambda x: -x[1])
        beams = nxt[:BEAM]
    return [tk for tk, _ in beams]


def run_game(g, quick):
    tr, va, te = lottery.split(g)
    if quick:
        te = te[:200]
    draws = tr + va + te
    a, b = len(tr), len(tr) + len(va)
    best = None
    for K in KS:
        m = HDC(g, K)
        online(m, draws, 0, a)
        rec = online(m, draws, a, b, keep=True)
        for beta in BETAS:
            bits = -sum(logp(beta, z)[i] for z, i in rec) / math.log(2)
            if best is None or bits < best[0]:
                best = (bits, K, beta, m)
    _, K, beta, m = best
    m.beta = beta
    out, top_hits = [], 0
    for t in range(b, len(draws)):
        hist = m.history(draws, t)
        tickets = beam(m, hist)
        bits = 0.0
        for j, v in enumerate(draws[t]):
            leg = g.legal(j, draws[t])
            bits -= logp(beta, m.z(j, m.cortex(hist, draws[t][:j]), leg))[leg.index(v)] / math.log(2)
        top_hits += draws[t] in tickets[:TOP]
        out.append((bits, tickets[0]))
        m.learn(draws, t, hist)
    r = lottery.judge(g, te, out)
    r["settings"] = {"K": K, "beta": beta}
    if g.kind == "digits":
        n, p = len(te), TOP * 10.0 ** -g.n
        r["top10"] = {"hits": top_hits, "expected": n * p, "p": lottery.binom_upper(top_hits, n, p)}
    return r


def main(quick):
    gs = lottery.games()
    gs = lottery.controls(len(gs[3].draws)) + gs
    results = {}
    for g in gs:
        t0 = time.time()
        r = run_game(g, quick)
        results[g.name] = r
        mo = r["money"]
        extra = f"  top-10 {r['top10']['hits']} vs {r['top10']['expected']:.1f} (p {r['top10']['p']:.3g})" \
            if "top10" in r else ""
        print(f"{g.name:<17} K={r['settings']['K']} beta={r['settings']['beta']:<5} "
              f"gain {r['gain_bits_per_draw']:+.4f} bits/draw  p {r['p']:.3g}  "
              f"halves {r['gain_first_half']:+.4f} {r['gain_second_half']:+.4f}  "
              f"ticket {mo['hits']} vs {mo['expected']:.1f} (p {mo['p']:.3g}){extra}  "
              f"{'PASS' if r['pass'] else 'fail'}  {time.time() - t0:.0f}s", flush=True)
    if not quick:
        json.dump(results, open(os.path.join(HERE, "results_hdc.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test: 200 test draws")
    main(ap.parse_args().quick)
