"""
BIG BRAIN — the flow brain on the full ada-haiku word base (68,298 glyphs)
=========================================================================
Same thinking as flow.py, same learning rule, on sparse matrices:

  pathways  27 sparse role matrices A_r[holder, value] from the shards;
            Ā_r = A_r split by the value's out-threads in its region (layer)
  streams   x_d = P x_{d-1},  P = Σ_r g_r Ā_r;  s_c = Σ_d γ^{d-1} x_d
  thought   score(w) = Σ_clues log(ε + s_c[w]); stops when the winner holds
            for 2 steps (own depth)
  learning  exact gradient of the answer's cross-entropy; local per thread
            (forward activity x backward error), as flow.py

Tests (answers are now ANY of 68,298 words — 170x more candidates than the
395-word brain):
  1. the 48 human questions (human_questions.py), written blind
  2. mixed-depth riddles (flow.py generator) on sampled words; pathway gains
     learned on train words, scored on disjoint test words

Run:  python big_brain.py
"""

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

import flow as FL
import human_questions as HQ
import reason as R
from glyph_schema3 import PATHS

HERE = Path(__file__).parent
LAYERS = ["grammar", "sentence", "idiom", "meaning"]


def load_base():
    G = {}
    for f in sorted((HERE / "ada_haiku" / "shards").glob("*.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            d = json.loads(line)
            G[d["word"]] = {p: e["value"] for p, e in d["glyph"].items()}
    words = sorted(G)
    leaves = sorted({v for g in G.values() for v in g.values()} - G.keys())
    names = words + leaves
    return SimpleNamespace(G=G, words=words, names=names,
                           idx={w: i for i, w in enumerate(names)}, N=len(names))


class BigFlow:
    def __init__(self, base, gamma=0.5, eps=1e-8, D=6):
        self.b, self.gamma, self.eps, self.D = base, gamma, eps, D
        self.n = len(base.words)
        idx = base.idx
        A = []
        for p in PATHS:
            rows, cols = [], []
            for w, g in base.G.items():
                if g[p] != w:
                    rows.append(idx[w]); cols.append(idx[g[p]])
            A.append(sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                                   shape=(base.N, base.N)))
        self.Abar = [None] * len(PATHS)
        for L in LAYERS:
            ks = [k for k, p in enumerate(PATHS) if p.startswith(L + ".")]
            out = np.asarray(sum(A[k] for k in ks).sum(0)).ravel()
            inv = sp.diags(np.where(out > 0, 1 / np.maximum(out, 1), 0).astype(np.float32))
            for k in ks:
                self.Abar[k] = (A[k] @ inv).tocsr()
        self.edges = [a.tocoo() for a in self.Abar]
        self.g = np.ones(len(PATHS))

    def P(self, g=None):
        g = self.g if g is None else g
        return sum(float(gr) * a for gr, a in zip(g, self.Abar)).tocsr()

    def streams(self, clues, D=None, g=None):
        D = D or self.D
        P = self.P(g)
        x = np.zeros((self.b.N, len(clues)), np.float32)
        for j, c in enumerate(clues):
            x[self.b.idx[c], j] = 1.0
        xs = [x]
        for _ in range(D):
            xs.append(np.asarray(P @ xs[-1], dtype=np.float32))
        return xs

    def accumulate(self, xs, D):
        return sum(self.gamma ** (d - 1) * xs[d] for d in range(1, D + 1))

    def score(self, s, cols, clues):
        sc = np.log(s[:self.n][:, cols].astype(np.float64) + self.eps).sum(1)
        for c in clues:
            if self.b.idx[c] < self.n:
                sc[self.b.idx[c]] = -np.inf
        return sc

    def think(self, qs, chunk=200):
        """[(top-3 words, rank of the answer, depth)] — own depth, as flow.py."""
        out = []
        for i in range(0, len(qs), chunk):
            part = qs[i:i + chunk]
            uniq = sorted({c for _, cl, _ in part for c in cl})
            col = {c: k for k, c in enumerate(uniq)}
            xs = self.streams(uniq)
            S = [None] + [self.accumulate(xs, d) for d in range(1, self.D + 1)]
            for t, cl, _ in part:
                cols = [col[c] for c in cl]
                prev, d = None, self.D
                for dd in range(1, self.D + 1):
                    w = int(np.argmax(self.score(S[dd], cols, cl)))
                    if w == prev:
                        d = dd; break
                    prev = w
                sc = self.score(S[d], cols, cl)
                top = [self.b.words[k] for k in np.argsort(-sc)[:3]]
                out.append((top, R.rank_of(sc, self.b.idx[t]), d))
        return out

    def loss_grad(self, qs, D=3, tau=1.0):
        uniq = sorted({c for _, cl, _ in qs for c in cl}); col = {c: k for k, c in enumerate(uniq)}
        xs = self.streams(uniq, D=D)
        s = self.accumulate(xs, D)
        P = self.P()
        loss, lam = 0.0, np.zeros_like(s)
        for t, cl, _ in qs:
            cols = [col[c] for c in cl]
            sc = self.score(s, cols, cl) / tau
            p = np.exp(sc - sc.max()); p /= p.sum()
            ti = self.b.idx[t]
            loss -= np.log(p[ti] + 1e-300)
            dsc = p.copy(); dsc[ti] -= 1.0
            for k in cols:
                lam[:self.n, k] += (dsc / tau / (s[:self.n, k] + self.eps)).astype(np.float32)
        lam /= len(qs)
        ys = [lam]
        PT = P.T.tocsr()
        for _ in range(D - 1):
            ys.append(np.asarray(PT @ ys[-1], dtype=np.float32))
        gr = np.zeros(len(PATHS))
        for r, e in enumerate(self.edges):
            for j in range(D):
                for k in range(D - j):
                    gr[r] += self.gamma ** (j + k) * float(
                        (e.data[:, None] * ys[j][e.row] * xs[k][e.col]).sum())
        return loss / len(qs), gr

    def train(self, qs, epochs=20, eta=0.5, batch=128, seed=0):
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            order = rng.permutation(len(qs))
            for i in range(0, len(qs), batch):
                _, gr = self.loss_grad([qs[j] for j in order[i:i + batch]])
                gr /= np.abs(gr).max() + 1e-12
                self.g = np.clip(self.g - eta * gr * self.g, 0.01, 100.0)
        return self


def lookup(base, qs):
    """CONTROL: count clue words in each word's own glyph."""
    inv = {}
    for k, w in enumerate(base.words):
        for v in set(base.G[w].values()):
            inv.setdefault(v, []).append(k)
    out = []
    for t, cl, _ in qs:
        s = np.zeros(len(base.words))
        for c in cl:
            for k in inv.get(c, []):
                s[k] += 1
        for c in cl:
            if c in base.G:
                s[base.idx[c]] = -1
        out.append(R.rank_of(s, base.idx[t]))
    return np.array(out)


def summary(name, ranks, extra=""):
    print(f"  {name:46} top-1 {np.mean(ranks < .5):6.1%}   top-5 {np.mean(ranks < 5):6.1%}   "
          f"top-20 {np.mean(ranks < 20):6.1%}   median rank {np.median(ranks) + 1:6.0f}{extra}")


if __name__ == "__main__":
    import time
    t0 = time.time()
    base = load_base()
    F = BigFlow(base)
    nnz = sum(a.nnz for a in F.Abar)
    print(f"big brain: {len(base.words):,} glyphed words + {base.N - len(base.words)} leaves, "
          f"{nnz:,} threads, built in {time.time() - t0:.0f}s\n")

    # ---- 1. human questions ------------------------------------------------
    qh, kept, dropped = HQ.prepare(base, HQ.QUESTIONS)
    print(f"1) HUMAN QUESTIONS: {len(qh)} questions; question words known {kept} "
          f"({kept / (kept + len(dropped)):.0%}; was 79% on the 395-word brain); "
          f"unknown: {', '.join(sorted(set(dropped))) or '-'}")
    print(f"   candidates: {len(base.words):,} words (was 395)")
    summary("lookup (question words in the answer's glyph)", lookup(base, qh))
    h0 = F.think(qh)
    summary("big brain, untrained", np.array([r for _, r, _ in h0]))

    # ---- 2. riddles + learning ---------------------------------------------
    rng = np.random.default_rng(0)
    pool = rng.permutation(base.words)
    train_w, test_w = list(pool[:900]), list(pool[900:1300])
    qtr = FL.mixed_riddles(base, train_w, per_word=2, seed=1)
    qte = FL.mixed_riddles(base, test_w, per_word=2, seed=2)
    print(f"\n2) MIXED-DEPTH RIDDLES (clues 1-3 hops): {len(qtr)} train / {len(qte)} test "
          f"on disjoint sampled words; candidates {len(base.words):,}")
    summary("lookup", lookup(base, qte))
    r0 = F.think(qte)
    summary("big brain, untrained", np.array([r for _, r, _ in r0]),
            f"   ({np.mean([d for *_, d in r0]):.1f} steps)")
    t1 = time.time()
    F.train(qtr)
    r1 = F.think(qte)
    summary("big brain, pathways learned", np.array([r for _, r, _ in r1]),
            f"   ({np.mean([d for *_, d in r1]):.1f} steps; trained in {time.time() - t1:.0f}s)")
    o = np.argsort(-F.g)
    print("   trusts: " + ", ".join(f"{PATHS[k].split('.')[-1]} {F.g[k]:.2f}" for k in o[:6]) +
          " | mutes: " + ", ".join(f"{PATHS[k].split('.')[-1]} {F.g[k]:.2f}" for k in o[-5:]))

    h1 = F.think(qh)
    print()
    summary("HUMAN QUESTIONS, big brain with learned pathways", np.array([r for _, r, _ in h1]))
    texts = dict(HQ.QUESTIONS)
    print()
    for (ans, cl, _), (top, r, d) in zip(qh, h1):
        mark = "HIT " if r < .5 else f"#{r + 1:<5.0f}"
        print(f"  {mark} {texts[ans]:62} -> {', '.join(top):38} (answer: {ans})")
