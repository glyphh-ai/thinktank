"""
PATHWAYS — a path becomes learned
=================================
The lattice's threads are innate wiring, shared by every brain. What a brain
learns from experience is which PATHS through them carry meaning: every thread
gets its own weight W_e, learned by the exact local rule

    ∂L/∂W_e = g_r · Ā_e · Σ_j γ^j · Σ_streams  y_j[target end] · X≤(D-1-j)[source end]

forward activity reaching the thread's source × error flowing back to its
target. That is nonzero only for threads on routes from the context words to
the answer, so credit flows along the paths that were used: a route that
carries the right answer again and again becomes a highway (myelination);
routes to wrong answers weaken; unused routes fade back toward innate (decay).

Same curriculum, controls and exams as curriculum.py (baby -> school on real
CHILDES / TinyStories / WordNet experience gated by age-of-acquisition; main,
replicate, shuffled-word control; the 48 human questions never trained on),
with more experience per stage — paths need repetition.

Run:  python pathways.py                                            # main
      python pathways.py --seed 100 --out pathways_results_rep      # replicate
      python pathways.py --shuffle --out pathways_results_shuffle   # control
"""

import argparse
import time

import numpy as np

import curriculum as CU
import big_brain as BB
import human_questions as HQ
from glyph_schema3 import PATHS

PER_STAGE = 30000          # utterances per stage (baby has only ~1,800 in total)


class PathLearner(CU.Learner):
    def __init__(self, base):
        super().__init__(base)
        self.w_min, self.w_max = 0.02, 50.0

    def step(self, qs, plast, D=3, eta=0.5, eta_w=0.1, decay=0.002):
        uniq = sorted({c for _, cl, _ in qs for c in cl}); col = {c: k for k, c in enumerate(uniq)}
        xs = self.streams(uniq, D=D)
        s = self.accumulate(xs, D)
        lam = np.zeros_like(s)
        loss = 0.0
        for t, cl, _ in qs:
            cols = [col[c] for c in cl]
            sc = self.score(s, cols, cl)
            p = np.exp(sc - sc.max()); p /= p.sum()
            ti = self.b.idx[t]
            loss -= np.log(p[ti] + 1e-300)
            dsc = p.copy(); dsc[ti] -= 1.0
            for k in cols:
                lam[:self.n, k] += (dsc / (s[:self.n, k] + self.eps)).astype(np.float32)
        lam /= len(qs)
        ys = [lam]
        PT = self.P().T.tocsr()
        for _ in range(D - 1):
            ys.append(np.asarray(PT @ ys[-1], dtype=np.float32))
        Xcum, acc = [], np.zeros_like(xs[0])                    # Σ_{k≤m} γ^k x_k
        for k in range(D):
            acc = acc + (self.gamma ** k) * xs[k]
            Xcum.append(acc)
        gW, g_role = [], np.zeros(len(PATHS))
        for r, (e, w) in enumerate(zip(self.edges, self.W)):
            corr = sum((self.gamma ** j) * (ys[j][e.row] * Xcum[D - 1 - j][e.col]).sum(1)
                       for j in range(D))                     # per thread: path credit
            gW.append(self.g[r] * e.data * corr)              # ∂L/∂W_e
            g_role[r] = float((e.data * w * corr).sum())      # ∂L/∂g_r
        pl = np.array([plast[CU.REGION[r]] for r in range(len(PATHS))])
        g_role /= np.abs(g_role).max() + 1e-12
        self.g = np.clip(self.g - eta * pl * g_role * self.g, 0.01, 100.0)
        gmax = max(float(np.abs(x).max()) for x in gW) + 1e-30
        for r in range(len(PATHS)):                           # learn the paths, let unused fade
            w = self.W[r] * np.exp(-eta_w * pl[r] * gW[r] / gmax)
            w = 1.0 + (w - 1.0) * (1.0 - decay)
            self.W[r] = np.clip(w, self.w_min, self.w_max).astype(np.float32)
        return loss / len(qs)

    def paths_report(self, k=12):
        names = self.b.names
        flat = []
        for r, (e, w) in enumerate(zip(self.edges, self.W)):
            for side in (np.argsort(-w)[:k], np.argsort(w)[:k]):
                for i in side:
                    flat.append((float(w[i]), f"{names[e.row[i]]} -{PATHS[r].split('.')[-1]}-> {names[e.col[i]]}"))
        flat.sort()
        return {"strongest": [(p, round(v, 2)) for v, p in flat[::-1][:k]],
                "weakest": [(p, round(v, 3)) for v, p in flat[:k]]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--out", default="pathways_results")
    ap.add_argument("--per-stage", type=int, default=PER_STAGE)
    args = ap.parse_args()
    CU.OUT = CU.HERE / args.out
    base = BB.load_base()
    CU.base_G = base.G
    aoa = CU.load_aoa()
    L = PathLearner(base)
    qh, _, _ = HQ.prepare(base, HQ.QUESTIONS)
    tag = f"{args.out}: "
    recs, done = CU.resume(L)
    if done:
        CU.log(f"resumed after: {', '.join(sorted(done, key=CU.STAGE_INDEX.get))}")
    else:
        human, hres = CU.exam(L, qh)
        recs.append(CU.record("newborn", L, None, human, hres, 0, 0, {"trained_on": 0}))
        CU.report(recs); CU.push(tag + "newborn exam"); CU.log(f"newborn: human {human}")

    for i, (name, source, ceiling, _, n_te, plast) in enumerate(CU.STAGES):
        if name in done:
            continue
        t0 = time.time()
        train, held, n_pool = CU.episodes(base, aoa, source, ceiling, args.per_stage, n_te,
                                          seed=args.seed + i)
        if args.shuffle:
            train = CU.shuffle_words(train, seed=1000 + i)
        nb = (len(train) + 127) // 128
        CU.log(f"{name}: {n_pool:,} real utterances qualify (AoA ≤ {ceiling}); training on {len(train):,}")
        for b in range(0, len(train), 128):
            loss = L.step(train[b:b + 128], plast)
            if (b // 128) % 20 == 0:
                CU.log(f"  {name} batch {b // 128 + 1}/{nb} loss {loss:.2f} "
                       f"({(time.time() - t0) / (b // 128 + 1):.0f}s/batch)")
        held_res, _ = CU.exam(L, held)
        human, hres = CU.exam(L, qh)
        rec = CU.record(name, L, held_res, human, hres, n_pool, time.time() - t0,
                        {"trained_on": len(train), "source": source, "aoa_ceiling": ceiling,
                         "plasticity": plast, "seed": args.seed, "shuffled": args.shuffle,
                         "paths": L.paths_report()})
        recs.append(rec)
        CU.report(recs)
        CU.push(tag + f"{name} stage: held-out top-1 {held_res['top1']:.1%}, human top-1 {human['top1']:.1%}")
        CU.log(f"{name}: held-out {held_res} | human {human} | regions {rec['region_gain']}")
    CU.log("pathways curriculum complete")
