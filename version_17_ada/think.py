"""
THINK — the lattice as an energy landscape: thinking settles, learning reshapes
===============================================================================
Structure (fixed, from the v3 glyphs):
    A_r[w, v] = 1  if word w's glyph has role r = v        27 sparse matrices
    Ŝ_r = D^-½ (A_r + A_rᵀ) D^-½                            symmetric, degree-
                                                            normalized (tames hubs)
The mind (learned): θ ∈ ℝ²⁷, one signed trust per role, init from its type
(attract/inherit/... +1/27, negate −1/27), kept in Σ|θ| ≤ 1.

Thinking — the lights x ∈ ℝᴺ settle to the minimum of a convex energy:
    M  = Σ_r θ_r Ŝ_r
    E(x) = ½ xᵀ(I − cM)x − (1−c)·bᵀx                     b = the question
    x_{t+1} = (1−c)·b + c·M·x_t   →   x* = (1−c)(I − cM)⁻¹ b
  ρ(cM) ≤ c·Σ|θ_r| < 1, so this is a contraction: thinking always settles,
  to a unique answer; the number of steps it takes is the effort.

Learning — equilibrium propagation (Scellier & Bengio 2017):
    free:   settle on the question                    → x*
    nudged: settle with input + β·(e_target − p)      → x̃
            (p = softmax of candidate lights: push the answer up, rivals down)
    Δθ_r = η · c/2 · ( x̃ᵀŜ_r x̃ − x*ᵀŜ_r x* ) / β
  Local: activity × activity on each thread, nudged minus free. Its gradient
  is checked against finite differences of the loss (--check).

Tasks (no answers stored anywhere):
  riddles   3 bare clue words from a word's glyph, roles NOT given; the
            lattice must settle on the word they describe. Words split
            train/test: test riddles are about words never practised.
  analogy   a : b :: c : ?   — find the role linking a to b, apply it to c
  multi-hop compose typed operators: similar(opposite(kind))

Run:  python think.py            # untrained vs trained, analogies, multi-hop
      python think.py --check    # EP gradient vs finite differences
"""

import argparse
import json
from pathlib import Path

import numpy as np

from glyph_schema3 import ROLE_PATHS, PATHS
from typed_encode import ROLE_TYPE

HERE = Path(__file__).parent
ETA_W = 0.05                            # per-thread step: the most active thread moves this much
SIGN = {"negate": -1.0}                 # every other type starts attracting


class Brain:
    def __init__(self, glyphs="glyphs3.jsonl", c=0.85):
        rows = [json.loads(l) for l in open(HERE / glyphs)]
        self.G = {d["word"]: {p: e["value"] for p, e in d["glyph"].items()} for d in rows}
        self.words = sorted(self.G)                                  # answerable words
        leaves = sorted({v for g in self.G.values() for v in g.values()} - self.G.keys())
        self.names = self.words + leaves
        self.idx = {w: i for i, w in enumerate(self.names)}
        self.N, self.c = len(self.names), c
        self.S = np.zeros((len(PATHS), self.N, self.N), np.float32)   # Ŝ_r
        self.A = np.zeros_like(self.S)                                # A_r (directed)
        for k, p in enumerate(PATHS):
            for w, g in self.G.items():
                if g[p] != w:
                    self.A[k, self.idx[w], self.idx[g[p]]] = 1.0
            S = self.A[k] + self.A[k].T
            d = S.sum(1)
            dn = np.where(d > 0, 1 / np.sqrt(np.maximum(d, 1e-9)), 0).astype(np.float32)
            self.S[k] = dn[:, None] * S * dn[None, :]
        self.edges = []                                               # Ŝ_r as (i, j, val)
        for k in range(len(PATHS)):
            i, j = np.nonzero(self.S[k])
            self.edges.append((i, j, self.S[k][i, j]))
        self.roles = [r for _, _, r in ROLE_PATHS]
        self.theta = np.array([SIGN.get(ROLE_TYPE[r], 1.0) for r in self.roles], np.float32)
        self.theta /= np.abs(self.theta).sum()
        self.cand = np.arange(len(self.words))                       # answers are glyphed words
        self.W = [np.ones(len(i), np.float32) for i, _, _ in self.edges]   # per-thread weights
        self.w_max = 1.15        # ρ(cM) ≤ c·Σ|θ|·w_max = 0.98 < 1: still a contraction

    # ---- thinking ----------------------------------------------------------
    def M(self, theta=None, W=None):
        """M = Σ_r θ_r (W_r ⊙ Ŝ_r): role trust × per-thread weight × structure."""
        th = self.theta if theta is None else theta
        W = self.W if W is None else W
        M = np.zeros((self.N, self.N), np.float64)
        for k, (i, j, v) in enumerate(self.edges):
            np.add.at(M, (i, j), th[k] * v * W[k])
        return M

    def settle(self, B, M=None, tol=1e-6, max_iter=500):
        """B: (N, batch) questions. Returns lights X* and the steps each took."""
        M = self.M() if M is None else M
        X = (1 - self.c) * B
        steps = np.zeros(B.shape[1], int)
        for t in range(1, max_iter + 1):
            Xn = (1 - self.c) * B + self.c * (M @ X)
            moving = np.abs(Xn - X).max(0) > tol
            steps[moving] = t
            X = Xn
            if not moving.any():
                break
        return X, steps

    def solve(self, B, M=None):
        """The settled state directly: x* = (1−c)(I − cM)⁻¹ B (same answer as settle)."""
        M = self.M() if M is None else M
        return np.linalg.solve(np.eye(self.N, dtype=np.float64) - self.c * M, (1 - self.c) * B)

    def quad(self, X):
        """xᵀ (W_r ⊙ Ŝ_r) x for every role r and question, over each role's edges only."""
        return np.stack([((v * w)[:, None] * X[i] * X[j]).sum(0)
                         for (i, j, v), w in zip(self.edges, self.W)])

    def edge_corr(self, X):
        """x_i x_j on every thread, averaged over the batch (the local activity product)."""
        return [(X[i] * X[j]).mean(1) for i, j, _ in self.edges]

    def question(self, clues):
        b = np.zeros(self.N, np.float32)
        for w in clues:
            b[self.idx[w]] = 1.0 / len(clues)
        return b

    def answer(self, X, exclude):
        s = X[self.cand].copy()
        for j, ex in enumerate(exclude):
            for w in ex:
                if w in self.idx and self.idx[w] < len(self.words):
                    s[self.idx[w], j] = -np.inf
        return s

    # ---- learning: equilibrium propagation --------------------------------
    def ep_grad(self, B, targets, exclude, beta=0.05, tau=0.01):
        L = np.eye(self.N) - self.c * self.M()
        X = np.linalg.solve(L, (1 - self.c) * B)
        s = self.answer(X, exclude) / tau
        p = np.exp(s - s.max(0)); p /= p.sum(0)
        nudge = -p
        nudge[targets, np.arange(len(targets))] += 1.0
        Bn = B.copy()
        Bn[self.cand] += beta * nudge / (1 - self.c)
        Xn = np.linalg.solve(L, (1 - self.c) * Bn)
        g_theta = self.c / 2 * (self.quad(Xn) - self.quad(X)).sum(1) / beta / B.shape[1]
        cn, cf = self.edge_corr(Xn), self.edge_corr(X)
        g_W = [self.c / 2 * self.theta[k] * v * (cn[k] - cf[k]) / beta
               for k, (_, _, v) in enumerate(self.edges)]
        return g_theta, g_W

    def loss(self, B, targets, exclude, theta=None, W=None, tau=0.01):
        X = self.solve(B, self.M(theta, W))
        s = self.answer(X, exclude) / tau
        s = s - s.max(0)
        return float(-(s[targets, np.arange(len(targets))] - np.log(np.exp(s).sum(0))).mean())

    def train(self, B, targets, exclude, epochs=30, eta=0.3, eta_w=0.0, batch=128, seed=0):
        """Equilibrium propagation on θ (role trust) and, if eta_w > 0, on every thread."""
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            order = rng.permutation(B.shape[1])
            for i in range(0, len(order), batch):
                j = order[i:i + batch]
                g, gW = self.ep_grad(B[:, j], targets[j], [exclude[k] for k in j])
                if eta:
                    self.theta += eta * g
                    n = np.abs(self.theta).sum()
                    if n > 1:
                        self.theta /= n
                if eta_w:                   # normalized: the most active thread moves eta_w
                    gmax = max(float(np.abs(g).max()) for g in gW) + 1e-12
                    for k in range(len(self.W)):
                        self.W[k] = np.clip(self.W[k] + eta_w * gW[k] / gmax, 0.0, self.w_max)
        return self


# ============================================================================
# Tasks
# ============================================================================
def riddles(brain, words, k=3, per_word=3, seed=0):
    rng = np.random.default_rng(seed)
    qs = []
    for w in words:
        vals = sorted({v for v in brain.G[w].values() if v != w})
        for _ in range(per_word):
            clues = list(rng.choice(vals, size=min(k, len(vals)), replace=False))
            qs.append((w, clues))
    B = np.stack([brain.question(c) for _, c in qs], 1)
    targets = np.array([brain.idx[w] for w, _ in qs])
    return B, targets, [c for _, c in qs], qs


def lookup(brain, qs):
    """CONTROL: no thinking — score each word by how many clues appear in its glyph."""
    vals = [set(brain.G[w].values()) for w in brain.words]
    hit = 0
    for w, clues in qs:
        score = np.array([len(set(clues) & v) + 1e-6 * k for k, v in enumerate(vals)], float)
        for c in clues:
            if c in brain.G:
                score[brain.idx[c]] = -1
        best = score.max()
        ties = np.flatnonzero(score >= best - 1e-3)        # random tie-break among equals
        hit += brain.idx[w] in ties and 1 / len(ties)
    return hit / len(qs)


def accuracy(brain, B, targets, exclude):
    X, steps = brain.settle(B)
    s = brain.answer(X, exclude)
    rank = (s > s[targets, np.arange(len(targets))]).sum(0)
    return float((rank == 0).mean()), float((rank < 5).mean()), steps


def analogy(brain, a, b, c):
    """a : b :: c : ?  — the role whose thread links a to b, applied to c."""
    ia, ib = brain.idx[a], brain.idx[b]
    links = [brain.roles[k] for k in range(len(PATHS)) if brain.A[k, ia, ib]]
    if not links or c not in brain.G:
        return links, None
    k = brain.roles.index(links[0])
    return links, brain.G[c][PATHS[k]]


def hop(brain, word, *roles):
    """Compose typed operators: x_{k+1} = A_{r_k} x_k, read the brightest light."""
    x = np.zeros(brain.N, np.float32); x[brain.idx[word]] = 1
    path = [word]
    for r in roles:
        x = brain.A[brain.roles.index(r)].T @ x
        if not x.any():
            break
        path.append(brain.names[int(np.argmax(x))])
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    brain = Brain()
    rng = np.random.default_rng(0)
    order = rng.permutation(brain.words)
    split = int(0.7 * len(order))
    train_w, test_w = list(order[:split]), list(order[split:])
    Btr, ttr, extr, _ = riddles(brain, train_w, seed=1)
    Bte, tte, exte, qte = riddles(brain, test_w, seed=2)
    print(f"brain: {brain.N} nodes ({len(brain.words)} glyphed), 27 role matrices, "
          f"c={brain.c}; riddles: {Btr.shape[1]} train / {Bte.shape[1]} test "
          f"(test words never practised)")

    if args.check:
        j = np.arange(16)
        ex = [extr[i] for i in j]
        g, gW = brain.ep_grad(Btr[:, j], ttr[j], ex, beta=1e-3)
        fd = np.zeros_like(brain.theta)
        for k in range(len(fd)):
            e = np.zeros_like(fd); e[k] = 1e-3
            fd[k] = -(brain.loss(Btr[:, j], ttr[j], ex, brain.theta + e)
                      - brain.loss(Btr[:, j], ttr[j], ex, brain.theta - e)) / 2e-3
        print(f"θ: EP vs finite-difference (−∂loss/∂θ): cosine "
              f"{g @ fd / (np.linalg.norm(g) * np.linalg.norm(fd)):.4f}")
        # per-thread weights: the threads the batch actually uses (largest |gradient|)
        cand = sorted(((abs(gW[k][e]), k, e) for k in range(len(gW))
                       for e in range(len(gW[k]))), reverse=True)[:24]
        ep_w, fd_w = [], []
        for _, k, e in cand:
            Wp = [w.copy() for w in brain.W]; Wm = [w.copy() for w in brain.W]
            # thread (i,j) and its mirror (j,i) share one weight: move both
            i0, j0, _ = brain.edges[k]
            mirror = np.flatnonzero((i0 == j0[e]) & (j0 == i0[e]))
            for m in [e, *mirror]:
                Wp[k][m] += 1e-3; Wm[k][m] -= 1e-3
            fd_w.append(-(brain.loss(Btr[:, j], ttr[j], ex, W=Wp)
                          - brain.loss(Btr[:, j], ttr[j], ex, W=Wm)) / 2e-3)
            ep_w.append(gW[k][e] + sum(gW[k][m] for m in mirror))
        ep_w, fd_w = np.array(ep_w), np.array(fd_w)
        print(f"per-thread W: EP vs finite-difference on the 24 busiest threads: cosine "
              f"{ep_w @ fd_w / (np.linalg.norm(ep_w) * np.linalg.norm(fd_w)):.4f}")
        raise SystemExit

    _, _, _, qtr = riddles(brain, train_w, seed=1)
    print(f"\nCONTROL lookup (no thinking: count clue overlap with each glyph): "
          f"test top-1 {lookup(brain, qte):.1%}")
    top1, top5, steps = accuracy(brain, Bte, tte, exte)
    print(f"UNTRAINED (θ from types only):        test top-1 {top1:.1%}  top-5 {top5:.1%}  "
          f"(chance {1 / len(brain.words):.2%})  settle steps {steps.mean():.1f}")
    theta0 = brain.theta.copy()
    brain.train(Btr, ttr, extr, eta=0.3, eta_w=0.0)
    t1, t5, _ = accuracy(brain, Bte, tte, exte); tr1, _, _ = accuracy(brain, Btr, ttr, extr)
    print(f"TRAINED θ only (27 numbers):          test top-1 {t1:.1%}  top-5 {t5:.1%}  "
          f"(train {tr1:.1%})")
    theta_r = brain.theta.copy()
    brain.train(Btr, ttr, extr, eta=0.0, eta_w=ETA_W, epochs=30)
    t1w, t5w, steps_w = accuracy(brain, Bte, tte, exte); tr1w, _, _ = accuracy(brain, Btr, ttr, extr)
    nW = sum(len(w) for w in brain.W)
    print(f"TRAINED + per-thread ({nW:,} weights): test top-1 {t1w:.1%}  top-5 {t5w:.1%}  "
          f"(train {tr1w:.1%})  settle steps {steps_w.mean():.1f}")
    allw = np.concatenate(brain.W)
    print(f"   threads: {np.mean(allw < 0.05):.1%} pruned (<0.05), {np.mean(allw > 1.1):.1%} "
          f"strengthened (>1.1), mean {allw.mean():.2f}")

    order_r = np.argsort(-theta_r)
    print("\nwhat it learned to trust (θ, init -> trained):")
    for k in list(order_r[:6]) + list(order_r[-4:]):
        print(f"   {brain.roles[k]:9} {theta0[k]:+.3f} -> {theta_r[k]:+.3f}")

    print("\na few test riddles (clues -> settled answer):")
    X, _ = brain.settle(Bte[:, :12])
    s = brain.answer(X, exte[:12])
    for j in range(0, 12, 2):
        w, clues = qte[j]
        got = brain.words[int(np.argmax(s[:, j]))]
        print(f"   {', '.join(clues):34} -> {got:12} {'OK' if got == w else '(was ' + w + ')'}")

    print("\neffort: clues given vs steps to settle and accuracy (test words)")
    for k in (1, 2, 3, 5, 8):
        B, t, ex, _ = riddles(brain, test_w, k=k, seed=3)
        a1, _, st = accuracy(brain, B, t, ex)
        print(f"   {k} clues: top-1 {a1:5.1%}   steps {st.mean():.1f}")

    print("\nanalogy  a : b :: c : ?")
    for a, b, c in (("love", "hate", "kind"), ("kind", "mean", "love"),
                    ("hate", "anger", "love"), ("love", "heart", "hate")):
        links, ans = analogy(brain, a, b, c)
        print(f"   {a} : {b} :: {c} : {ans}     (link found: {', '.join(links) or 'none'})")

    print("\nmulti-hop (composed typed operators)")
    for w, rs in (("kind", ("opposite", "similar")), ("love", ("cause", "opposite")),
                  ("hate", ("opposite", "purpose"))):
        print(f"   {' -> '.join(rs)} of {w}:  {' -> '.join(hop(brain, w, *rs))}")
