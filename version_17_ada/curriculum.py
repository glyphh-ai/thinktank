"""
CURRICULUM — raise the big brain from baby to school on real human experience
=============================================================================
Organic: no designed task, no tuning on any exam, nothing hand-picked.

  experience  real utterances, in the order humans meet them
                baby     CHILDES speech (children + caregivers), every word
                         learned by age 3.5 (Kuperman 2012 age-of-acquisition)
                toddler  CHILDES, words learned by 5
                child    TinyStories sentences, words learned by 7
                school   WordNet definitions -> the word they define, words by 11
              One word of each utterance is held out; the brain thinks from the
              others (prediction — how children learn words from context).
  learning    the same exact local rule as flow.py (forward activity x
              backward error per thread) on the 27 pathway gains,
              + FAMILIARITY: every thread strengthens with the energy it
                carries (practice), so words it hears often grow strong
  regions     EVERY region learns from the first day; how strongly each is
              reinforced grows by stage (fixed below, before any run):

                       meaning  sentence  idiom  grammar
                baby     1.0      0.3      0.2     0.1
                toddler  1.0      1.0      0.3     0.2
                child    1.0      1.0      1.0     0.5
                school   1.0      1.0      1.0     1.5   (reinforced later, harder)

  exams       after EVERY stage (never trained on):
                · the 48 human questions (human_questions.py, written blind)
                · held-out utterances of that stage
                · what it trusts / mutes, by region; sample thoughts
              results -> curriculum_results/ (pushed after every stage)

Data (not committed; data/dev/):  en.aoa.csv (Kuperman et al. 2012, via
HuggingFace StephanAkkerman/English-Age-of-Acquisition), childes_train.txt
(CHILDES via HuggingFace w-nicole/childes_data_no_tags).

Run:  python curriculum.py
"""

import csv
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import big_brain as BB
import human_questions as HQ
from glyph_schema3 import PATHS

HERE = Path(__file__).parent
DATA = HERE / "data" / "dev"
OUT = HERE / "curriculum_results"
LAYERS = ["grammar", "sentence", "idiom", "meaning"]
REGION = [p.split(".")[0] for p in PATHS]

STAGES = [   # name, source, AoA ceiling, train episodes, held-out episodes, plasticity
    ("baby", "childes", 3.5, 3000, 300, {"meaning": 1.0, "sentence": 0.3, "idiom": 0.2, "grammar": 0.1}),
    ("toddler", "childes", 5.0, 3000, 300, {"meaning": 1.0, "sentence": 1.0, "idiom": 0.3, "grammar": 0.2}),
    ("child", "tinystories", 7.0, 3000, 300, {"meaning": 1.0, "sentence": 1.0, "idiom": 1.0, "grammar": 0.5}),
    ("school", "wordnet", 11.0, 3000, 300, {"meaning": 1.0, "sentence": 1.0, "idiom": 1.0, "grammar": 1.5}),
]
TOK = re.compile(r"[a-z]+")


# ============================================================================
# experience
# ============================================================================
def load_aoa():
    aoa = {}
    with open(DATA / "en.aoa.csv") as f:
        for row in csv.DictReader(f):
            v = row["AoA_Kup_lem"] or row["AoA_Kup"]
            try:
                aoa[row["Word"].lower()] = float(v)
            except ValueError:
                pass
    return aoa


def utterances(source):
    if source == "childes":
        for line in open(DATA / "childes_train.txt"):
            yield re.sub(r"^\[\w+\]\s*", "", line.strip().lower()), None
    elif source == "tinystories":
        text = open(HERE.parent / "version_16_base/data/tinystories_train_100mb.txt").read().lower()
        for s in re.split(r"[.!?\n]+", text):
            yield s, None
    elif source == "wordnet":
        from nltk.corpus import wordnet as wn
        for syn in wn.all_synsets():
            for lem in syn.lemma_names():
                if re.fullmatch(r"[a-z]+", lem):
                    yield syn.definition().lower(), lem


def episodes(base, aoa, source, ceiling, n_train, n_test, seed):
    """[(target, clues, _)] from real utterances whose every content word is known by `ceiling`."""
    rng = np.random.default_rng(seed)
    pool = []
    for text, target in utterances(source):
        words = [w for w in TOK.findall(text) if w not in HQ.FILLER]
        if not words or any(w not in base.idx or aoa.get(w, 99) > ceiling for w in words):
            continue
        if target is None:                                   # hold out one word of the utterance
            cands = [w for w in dict.fromkeys(words) if w in base.G]
            if len(set(words)) < 3 or not cands:
                continue
            target = cands[rng.integers(len(cands))]
        elif target not in base.G or aoa.get(target, 99) > ceiling:
            continue
        clues = [w for w in dict.fromkeys(words) if w != target]
        if len(clues) >= 2:
            pool.append((target, clues, [0] * len(clues)))
    order = rng.permutation(len(pool))
    pool = [pool[i] for i in order]
    n_test = min(n_test, len(pool) // 10)                   # small stages: hold out 10%
    return pool[n_test:n_test + n_train], pool[:n_test], len(pool)


# ============================================================================
# the learner: big brain + familiarity + region plasticity
# ============================================================================
class Learner(BB.BigFlow):
    def __init__(self, base):
        super().__init__(base)
        self.W = [np.ones(e.nnz, np.float32) for e in self.edges]        # familiarity per thread
        self.w_max = 5.0

    def P(self, g=None):
        g = self.g if g is None else g
        mats = [sp.csr_matrix((e.data * w * float(gr), (e.row, e.col)), shape=e.shape)
                for e, w, gr in zip(self.edges, self.W, g)]
        return sum(mats).tocsr()

    def step(self, qs, plast, D=3, eta=0.5, eta_f=0.2):
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
        gr = np.zeros(len(PATHS)); use = []
        for r, (e, w) in enumerate(zip(self.edges, self.W)):
            a = e.data * w
            for j in range(D):
                for k in range(D - j):
                    gr[r] += self.gamma ** (j + k) * float((a[:, None] * ys[j][e.row] * xs[k][e.col]).sum())
            use.append(sum((a[:, None] * self.g[r] * xs[k][e.col]).sum(1) for k in range(D)))
        pl = np.array([plast[REGION[r]] for r in range(len(PATHS))])
        gr /= np.abs(gr).max() + 1e-12                                     # exact gradient, normalized
        self.g = np.clip(self.g - eta * pl * gr * self.g, 0.01, 100.0)
        umax = max(float(u.max()) for u in use) + 1e-12                    # practice -> familiarity
        for r, u in enumerate(use):
            self.W[r] = np.clip(self.W[r] + eta_f * pl[r] * u / umax, 1.0, self.w_max)
        return loss / len(qs)


# ============================================================================
# exams + record
# ============================================================================
def exam(L, qs):
    res = L.think(qs)
    r = np.array([x[1] for x in res])
    return {"n": len(qs), "top1": float(np.mean(r < .5)), "top5": float(np.mean(r < 5)),
            "top20": float(np.mean(r < 20)), "median_rank": float(np.median(r) + 1)}, res


def record(stage, L, held, human, human_res, n_pool, secs, extra=None):
    o = np.argsort(-L.g)
    regions = {Lr: float(np.mean([L.g[k] for k in range(len(PATHS)) if REGION[k] == Lr]))
               for Lr in LAYERS}
    allW = np.concatenate(L.W)
    texts = dict(HQ.QUESTIONS)
    qh, _, _ = HQ.prepare(L.b, HQ.QUESTIONS)
    rec = {"stage": stage, "episodes_available": n_pool, "seconds": round(secs),
           "held_out": held, "human_questions": human,
           "region_gain": regions,
           "trusts": [(PATHS[k].split(".")[-1], round(float(L.g[k]), 2)) for k in o[:6]],
           "mutes": [(PATHS[k].split(".")[-1], round(float(L.g[k]), 2)) for k in o[-5:]],
           "familiarity": {"mean": float(allW.mean()), "strengthened_>1.5": float(np.mean(allW > 1.5))},
           "human_answers": [{"q": texts[a], "answer": a, "top3": top, "rank": float(r + 1)}
                             for (a, _, _), (top, r, _) in zip(qh, human_res)]}
    if extra:
        rec.update(extra)
    OUT.mkdir(exist_ok=True)
    (OUT / f"{len(list(OUT.glob('*.json'))):02d}_{stage}.json").write_text(json.dumps(rec, indent=1))
    return rec


def report(recs):
    lines = ["# Curriculum — baby to school (organic run)", "",
             "Same 48 human questions after every stage (never trained on). "
             "68,298 candidate answers.", "",
             "| stage | episodes | held-out top-1 / top-5 | human Q top-1 / top-5 / top-20 | median rank | "
             "region gains (meaning / sentence / idiom / grammar) |",
             "|---|---|---|---|---|---|"]
    for r in recs:
        h, q, g = r["held_out"], r["human_questions"], r["region_gain"]
        held = f"{h['top1']:.1%} / {h['top5']:.1%}" if h else "-"
        lines.append(f"| {r['stage']} | {r.get('trained_on', 0):,} | {held} | {q['top1']:.1%} / "
                     f"{q['top5']:.1%} / {q['top20']:.1%} | {q['median_rank']:.0f} | "
                     f"{g['meaning']:.2f} / {g['sentence']:.2f} / {g['idiom']:.2f} / {g['grammar']:.2f} |")
    lines += ["", "## Latest thoughts on the human questions", ""]
    for a in recs[-1]["human_answers"]:
        mark = "HIT" if a["rank"] <= 1 else f"#{a['rank']:.0f}"
        lines.append(f"- {mark} — *{a['q']}* → {', '.join(a['top3'])} (answer: {a['answer']})")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def push(msg):
    root = HERE.parent
    subprocess.run(["git", "-C", str(root), "add", str(OUT)], capture_output=True)
    r = subprocess.run(["git", "-C", str(root), "commit", "-q", "-m",
                        f"curriculum: {msg}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
                        "Claude-Session: https://claude.ai/code/session_01VUsxdcdhXYHqLqZHQQHBh4"],
                       capture_output=True)
    if r.returncode == 0:
        for d in (0, 2, 4, 8):
            time.sleep(d)
            if subprocess.run(["git", "-C", str(root), "push", "-q", "origin", "HEAD"],
                              capture_output=True).returncode == 0:
                break


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


if __name__ == "__main__":
    base = BB.load_base()
    aoa = load_aoa()
    L = Learner(base)
    qh, _, _ = HQ.prepare(base, HQ.QUESTIONS)
    log(f"brain: {len(base.words):,} words; AoA norms for {len(aoa):,} words")

    recs = []
    t0 = time.time()
    human, hres = exam(L, qh)
    recs.append(record("newborn", L, None, human, hres, 0, 0, {"trained_on": 0}))
    report(recs); push("newborn exam"); log(f"newborn: human {human}")

    for i, (name, source, ceiling, n_tr, n_te, plast) in enumerate(STAGES):
        t0 = time.time()
        train, held, n_pool = episodes(base, aoa, source, ceiling, n_tr, n_te, seed=10 + i)
        log(f"{name}: {n_pool:,} real utterances qualify (AoA ≤ {ceiling}); training on {len(train):,}")
        for b in range(0, len(train), 128):
            loss = L.step(train[b:b + 128], plast)
            if (b // 128) % 5 == 0:
                log(f"  {name} batch {b // 128 + 1}/{(len(train) + 127) // 128} loss {loss:.2f}")
        held_res, _ = exam(L, held)
        human, hres = exam(L, qh)
        rec = record(name, L, held_res, human, hres, n_pool, time.time() - t0,
                     {"trained_on": len(train), "source": source, "aoa_ceiling": ceiling,
                      "plasticity": plast,
                      "sample_experience": [" ".join(c) + f"  ->  {t}" for t, c, _ in train[:8]]})
        recs.append(rec)
        report(recs)
        push(f"{name} stage: held-out top-1 {held_res['top1']:.1%}, human top-1 {human['top1']:.1%}")
        log(f"{name}: held-out {held_res} | human {human} | regions {rec['region_gain']}")
    log("curriculum complete")
