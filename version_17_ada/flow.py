"""
FLOW — the brain finds its own depth, and learns its pathways
=============================================================
brain.py, extended:

  depth     each clue's stream keeps flowing, older steps fading:
                s_c = Σ_{d=1..D} γ^{d-1} x_d,      x_d = P x_{d-1}
            and the thought STOPS when the winner stops changing (same top
            answer for 2 consecutive steps). Nobody tells it how deep to go.
  pathways  P = Σ_r g_r Ā_r — one learnable gain per role pathway (27),
            Ā_r = A_r split by the source's out-threads in its region
            (as brain.py). Directed: value -> holder.
  thought   score(w) = Σ_clues log(ε + s_c[w])      coincidence of streams
  learning  exact gradient of the answer's cross-entropy. The flow is linear,
            so for each pathway r it is LOCAL:
                ∂L/∂g_r = Σ_{j+k<D} γ^{j+k} Σ_{threads v→w of r} Ā_r[w,v] · y_j[w] · x_k[v]
            x_k = the forward signal k steps out of the clue (activity at the
            source), y_j = the error flowing backward j steps from the answer
            (activity at the target) — activity × error on every thread.
            Checked against finite differences (--check).

Task: MIXED-DEPTH riddles — each clue sits 1, 2 or 3 hops from the answer
(true distance: not reachable in fewer hops), so no fixed depth fits.
Baseline: explicit search up to 3 hops (count clues within 3 hops).

Run:  python flow.py           # depth and learning, verified on unseen words
      python flow.py --check   # gradient vs finite differences
"""

import argparse

import numpy as np

import reason as R
import think as T
from glyph_schema3 import PATHS

LAYERS = ["grammar", "sentence", "idiom", "meaning"]


# ============================================================================
# The brain
# ============================================================================
class Flow:
    def __init__(self, base, gamma=0.5, eps=1e-8, D=6):
        self.b, self.gamma, self.eps, self.D = base, gamma, eps, D
        self.n = len(base.words)
        self.Abar = []
        for L in LAYERS:
            ks = [k for k, p in enumerate(PATHS) if p.startswith(L + ".")]
            out = sum(base.A[k] for k in ks).sum(0)
            for k in ks:
                self.Abar.append((k, base.A[k] / np.where(out > 0, out, 1)[None, :]))
        self.Abar.sort()
        self.Abar = [a for _, a in self.Abar]                  # indexed like PATHS
        self.edges = [np.nonzero(a) for a in self.Abar]
        self.g = np.ones(len(PATHS))

    def P(self, g=None):
        g = self.g if g is None else g
        return sum(gr * a for gr, a in zip(g, self.Abar))

    def streams(self, clues, D=None, g=None):
        """Forward signals x_1..x_D for each clue (N × clues each)."""
        D = D or self.D
        P = self.P(g)
        x = np.zeros((self.b.N, len(clues)))
        for j, c in enumerate(clues):
            x[self.b.idx[c], j] = 1.0
        xs = [x]
        for _ in range(D):
            xs.append(P @ xs[-1])
        return xs                                               # xs[0] = the clues themselves

    def accumulate(self, xs, D):
        return sum(self.gamma ** (d - 1) * xs[d] for d in range(1, D + 1))

    def score(self, s, clue_cols, clues):
        sc = np.log(s[:self.n][:, clue_cols] + self.eps).sum(1)
        for c in clues:
            if self.b.idx[c] < self.n:
                sc[self.b.idx[c]] = -np.inf
        return sc

    # ---- thinking with its own depth --------------------------------------
    def think(self, qs, fixed=None, g=None):
        """Ranks of the answers; depth fixed, or chosen by the thought itself."""
        uniq = sorted({c for _, cl, _ in qs for c in cl})
        col = {c: i for i, c in enumerate(uniq)}
        xs = self.streams(uniq, g=g)
        S = [None] + [self.accumulate(xs, d) for d in range(1, self.D + 1)]
        ranks, depths = [], []
        for t, cl, _ in qs:
            cols = [col[c] for c in cl]
            if fixed:
                d = fixed
            else:                                   # stop when the winner holds for 2 steps
                prev, d = None, self.D
                for dd in range(1, self.D + 1):
                    w = int(np.argmax(self.score(S[dd], cols, cl)))
                    if w == prev:
                        d = dd
                        break
                    prev = w
            ranks.append(R.rank_of(self.score(S[d], cols, cl), self.b.idx[t]))
            depths.append(d)
        return np.array(ranks), np.array(depths)

    # ---- learning: exact, local gradient -----------------------------------
    def loss_grad(self, qs, D=3, tau=1.0, g=None, grad=True):
        uniq = sorted({c for _, cl, _ in qs for c in cl})
        col = {c: i for i, c in enumerate(uniq)}
        g = self.g if g is None else g
        xs = self.streams(uniq, D=D, g=g)
        s = self.accumulate(xs, D)
        P = self.P(g)
        loss, lam = 0.0, np.zeros_like(s)                       # λ = ∂L/∂s (per clue column)
        for t, cl, _ in qs:
            cols = [col[c] for c in cl]
            sc = self.score(s, cols, cl) / tau
            p = np.exp(sc - sc.max()); p /= p.sum()
            ti = self.b.idx[t]
            loss -= np.log(p[ti] + 1e-300)
            dsc = p.copy(); dsc[ti] -= 1.0                     # ∂L/∂score
            for c, k in zip(cl, cols):
                lam[:self.n, k] += dsc / tau / (s[:self.n, k] + self.eps)
        loss /= len(qs); lam /= len(qs)
        if not grad:
            return loss
        ys = [lam]                                              # error flowing backward
        for _ in range(D - 1):
            ys.append(P.T @ ys[-1])
        gr = np.zeros(len(PATHS))
        for r, (wi, vi) in enumerate(self.edges):
            a = self.Abar[r][wi, vi]
            for j in range(D):
                for k in range(D - j):                          # d = j + k + 1 ≤ D
                    gr[r] += self.gamma ** (j + k) * (a[:, None] * ys[j][wi] * xs[k][vi]).sum()
        return loss, gr

    def train(self, qs, epochs=40, eta=0.5, batch=256, D=3, seed=0):
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = rng.permutation(len(qs))
            for i in range(0, len(qs), batch):
                _, gr = self.loss_grad([qs[j] for j in order[i:i + batch]], D=D)
                gr /= np.abs(gr).max() + 1e-12                  # normalized step
                self.g = np.clip(self.g - eta * gr * self.g, 0.01, 100.0)   # multiplicative
        return self


# ============================================================================
# Mixed-depth riddles
# ============================================================================
def hood(base, w, d):
    """Words reachable from w in ≤ d hops along glyph values (out-direction)."""
    seen, front = {w}, {w}
    for _ in range(d):
        front = {v for u in front if u in base.G for v in base.G[u].values()} - seen
        seen |= front
    return seen


def mixed_riddles(base, words, k=3, per_word=6, depths=(1, 2, 3), seed=0):
    rng = np.random.default_rng(seed)
    qs = []
    for t in words:
        for _ in range(per_word * 4):
            if sum(q[0] == t for q in qs) >= per_word:
                break
            firsts, clues, ds = set(), [], []
            for d in rng.choice(depths, size=k):
                path = [t]
                for _ in range(d):
                    vals = sorted({v for v in base.G.get(path[-1], {}).values()} - set(path))
                    if not vals:
                        break
                    path.append(str(rng.choice(vals)))
                if len(path) != d + 1 or path[1] in firsts:
                    continue
                c = path[-1]
                if c in hood(base, t, d - 1) or c in clues or c not in base.idx:
                    continue                                    # true distance must be d
                firsts.add(path[1]); clues.append(c); ds.append(int(d))
            if len(clues) == k:
                qs.append((t, clues, ds))
    return qs


def search_ranks(base, qs, d=3):
    """Explicit search: count the clues found within d hops of each word."""
    H = [hood(base, w, d) for w in base.words]
    out = []
    for t, cl, _ in qs:
        s = np.array([sum(c in h for c in cl) for h in H], float)
        for c in cl:
            if c in base.G:
                s[base.idx[c]] = -1
        out.append(R.rank_of(s, base.idx[t]))
    return np.array(out)


def show(name, r, extra=""):
    print(f"  {name:34} top-1 {np.mean(r < .5):6.1%}   top-5 {np.mean(r < 5):6.1%}   "
          f"median rank {np.median(r):5.0f}{extra}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    base = T.Brain()
    F = Flow(base)
    rng = np.random.default_rng(0)
    order = rng.permutation(base.words)
    split = int(0.7 * len(order))
    train_w, test_w = list(order[:split]), list(order[split:])
    qtr = mixed_riddles(base, train_w, seed=1)
    qte = mixed_riddles(base, test_w, seed=2)

    if args.check:
        qs = qtr[:24]
        F.g = np.random.default_rng(3).uniform(0.5, 1.5, len(PATHS))
        _, gr = F.loss_grad(qs)
        fd = np.zeros(len(PATHS))
        for r in range(len(PATHS)):
            e = np.zeros(len(PATHS)); e[r] = 1e-5
            fd[r] = (F.loss_grad(qs, g=F.g + e, grad=False)
                     - F.loss_grad(qs, g=F.g - e, grad=False)) / 2e-5
        print(f"pathway gradient (activity x error) vs finite differences: cosine "
              f"{gr @ fd / (np.linalg.norm(gr) * np.linalg.norm(fd)):.6f}, "
              f"max rel err {np.max(np.abs(gr - fd)) / np.max(np.abs(fd)):.2e}")
        raise SystemExit

    mix = {d: sum(dd.count(d) for _, _, dd in qte) for d in (1, 2, 3)}
    print(f"mixed-depth riddles: {len(qtr)} train / {len(qte)} test (unseen answers); "
          f"test clue depths {mix}; chance {1 / len(base.words):.2%}\n")

    print("1) DEPTH")
    show("explicit search, ≤3 hops", search_ranks(base, qte, 3))
    for d in (1, 2, 3):
        show(f"brain, told depth {d}", F.think(qte, fixed=d)[0])
    r_own, dep = F.think(qte)
    show("brain, finds its own depth", r_own,
         f"   (stops at {dep.mean():.1f} steps; " +
         ", ".join(f"{d}:{np.mean(dep == d):.0%}" for d in range(1, F.D + 1) if np.mean(dep == d)) + ")")

    print("\n2) LEARNING (pathway gains, exact local gradient, trained on train words only)")
    r_tr0 = F.think(qtr)[0]
    F.train(qtr)
    r_tr1 = F.think(qtr)[0]
    r_le, dep2 = F.think(qte)
    show("train words, before", r_tr0)
    show("train words, after", r_tr1)
    show("UNSEEN words, before", r_own)
    show("UNSEEN words, after", r_le, f"   (stops at {dep2.mean():.1f} steps)")
    o = np.argsort(-F.g)
    print("  pathways it learned to trust:  " +
          ", ".join(f"{PATHS[k].split('.')[-1]} {F.g[k]:.2f}" for k in o[:7]))
    print("  pathways it learned to mute:   " +
          ", ".join(f"{PATHS[k].split('.')[-1]} {F.g[k]:.2f}" for k in o[-6:]))
    region = {L: np.mean([F.g[k] for k, p in enumerate(PATHS) if p.startswith(L + ".")])
              for L in LAYERS}
    print("  mean gain by region:           " + ", ".join(f"{L} {v:.2f}" for L, v in region.items()))
