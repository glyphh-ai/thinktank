"""
SOUND GLYPHS — sound has the same structure as a word glyph, with different features
====================================================================================
A heard sound becomes a glyph: one layer ("sound"), segments, roles, and every
role filled by exactly one value. Every sound glyph has the SAME atomic makeup
(15 roles, unweighted, no "none"), just like every word glyph has 27.

  layer sound
    source   pitch      low / high            (median voice pitch)
             contour    falling / flat / rising (pitch slope)
             voicing    whisper / voiced      (how much of it is voiced)
    energy   loudness   quiet / loud          (peak level)
             attack     sudden / gentle       (time to reach the peak)
             decay      steady / fading       (how much it dies away)
    time     duration   short / long
             tempo      slow / fast           (energy beats per second)
             rhythm     even / uneven         (how jumpy the envelope is)
    timbre   brightness dark / bright         (spectral centroid)
             roughness  smooth / harsh        (spectral flux)
             noise      clear / noisy         (spectral flatness)
    form     onset      sound atom            (the learned primitive heard first,
             nucleus    sound atom             in the middle,
             coda       sound atom             and last)

The value of a feature role is a WORD that already has a glyph in the 68,298-word
lattice. The split points come from the heard training sounds only (medians /
terciles; no labels). The form roles hold the sound primitives learned without
labels in sound.py (64 atoms). Each value is typed by its role (loud-as-loudness).

Hearing fires the sound glyph into the SAME brain:
  - its feature value words light up in the lattice (loud, harsh, ...) and that
    energy spreads through the lattice like any other thought;
  - each typed value travels learned paths to words (same exact local rule as
    sound.py and pathways.py).

Same data, same held-out speakers, same stages (one syllable, then two), same
shuffled-label control as the primitives-only baseline (sound.py).

Also a probe, NOT trained on: what the lattice lights up for loud/harsh/sudden
sounds versus quiet/smooth/gentle ones (emotion words). This is innate lattice
wiring only; emotion as a learned layer comes later.

Run:  python sound_glyph.py
"""

import json
import time
from pathlib import Path

import numpy as np

import big_brain as BB
import sound as S

HERE = Path(__file__).parent

ROLES = [("source", "pitch", ["low", "high"]),
         ("source", "contour", ["falling", "flat", "rising"]),
         ("source", "voicing", ["whisper", "voiced"]),
         ("energy", "loudness", ["quiet", "loud"]),
         ("energy", "attack", ["sudden", "gentle"]),
         ("energy", "decay", ["steady", "fading"]),
         ("time", "duration", ["short", "long"]),
         ("time", "tempo", ["slow", "fast"]),
         ("time", "rhythm", ["even", "uneven"]),
         ("timbre", "brightness", ["dark", "bright"]),
         ("timbre", "roughness", ["smooth", "harsh"]),
         ("timbre", "noise", ["clear", "noisy"])]
FORM = ["onset", "nucleus", "coda"]
HOT = ["anger", "angry", "aggressive", "excited", "fear", "scared", "surprise"]
CALM = ["calm", "peace", "comfort", "relaxed", "safe", "tender"]


# ============================================================================
# acoustic features from the raw waveform
# ============================================================================
def features(x):
    """waveform (int16) -> 12 raw feature numbers (in ROLES order)."""
    x = x.astype(np.float64) / 32768
    fr = np.lib.stride_tricks.sliding_window_view(x, 400)[::160]
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    on = np.where(db > db.max() - 25)[0]
    a, b = on[0], on[-1] + 1
    fr, env = fr[a:b], db[a:b]
    T = len(env)
    win = fr * np.hanning(400)
    spec = np.abs(np.fft.rfft(win, axis=1)) ** 2 + 1e-12
    # pitch by autocorrelation (80-400 Hz)
    acf = np.fft.irfft(np.abs(np.fft.rfft(win, 800, axis=1)) ** 2, axis=1)[:, :201]
    r = acf[:, 40:201] / (acf[:, :1] + 1e-12)
    lag = r.argmax(1) + 40
    voiced = r.max(1) > 0.5
    f0 = 16000 / lag
    pitch = np.median(f0[voiced]) if voiced.any() else 150.0
    if voiced.sum() >= 3:
        t = np.where(voiced)[0]
        contour = np.polyfit(t / 100, np.log(f0[voiced]), 1)[0]
    else:
        contour = 0.0
    voicing = voiced.mean()
    loudness = db.max()
    pk = int(env.argmax())
    attack = pk
    tail = env[max(pk, T - max(T // 4, 1)):]
    decay = env.max() - tail.mean()
    duration = T
    sm = np.convolve(env, np.ones(5) / 5, mode="same")
    peaks = [i for i in range(1, T - 1) if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1]
             and sm[i] > sm.max() - 12]
    tempo = len(peaks) / (T / 100)
    rhythm = np.abs(np.diff(env, 2)).mean() if T > 2 else 0.0
    freqs = np.fft.rfftfreq(400, 1 / 16000)
    brightness = ((spec * freqs).sum(1) / spec.sum(1)).mean()
    ns = spec / np.linalg.norm(spec, axis=1, keepdims=True)
    roughness = np.linalg.norm(np.diff(ns, axis=0), axis=1).mean() if T > 1 else 0.0
    noise = (np.exp(np.log(spec).mean(1)) / spec.mean(1)).mean()
    return np.array([pitch, contour, voicing, loudness, attack, decay, duration,
                     tempo, rhythm, brightness, roughness, noise])


def form(frames, C):
    """the primitive (atom) most heard in each third: onset, nucleus, coda."""
    d = (frames ** 2).sum(1)[:, None] - 2 * frames @ C.T + (C ** 2).sum(1)[None]
    seq = d.argmin(1)
    parts = np.array_split(seq, 3) if len(seq) >= 3 else [seq, seq, seq]
    return [int(np.bincount(p, minlength=S.K).argmax()) for p in parts]


class Glypher:
    """Split points learned from heard training sounds (no labels)."""

    def __init__(self, F):
        self.cuts = [np.quantile(F[:, k], np.linspace(0, 1, len(v) + 1)[1:-1])
                     for k, (_, _, v) in enumerate(ROLES)]

    def glyph(self, f, atoms):
        g = {}
        for k, ((seg, role, vals), c) in enumerate(zip(ROLES, self.cuts)):
            g[f"sound.{seg}.{role}"] = vals[int(np.searchsorted(c, f[k]))]
        for role, a in zip(FORM, atoms):
            g[f"sound.form.{role}"] = f"atom{a}"
        return g


# typed nodes: one per (role, value)
NODES = [f"sound.{s}.{r}={v}" for s, r, vs in ROLES for v in vs] + \
        [f"sound.form.{r}=atom{a}" for r in FORM for a in range(S.K)]
NODE = {n: i for i, n in enumerate(NODES)}


def act(g):
    """a sound glyph -> node activity (15 values, unweighted)."""
    return {NODE[f"{p}={v}"]: 1.0 / len(g) for p, v in g.items()}


# ============================================================================
# the sound-glyph region of the same brain
# ============================================================================
class GlyphHearing(S.Hearing):
    def __init__(self, brain, words, inject=True):
        self.F, self.b = brain, brain.b
        self.n_sound = len(NODES)
        self.words = list(words)
        self.widx = np.array([self.b.idx[w] for w in self.words])
        self.V = np.zeros((self.n_sound, len(self.words)), np.float32)
        self.formed = np.zeros_like(self.V, dtype=bool)
        self.inject = inject
        # feature value words fire straight into the lattice
        self.inj = np.array([self.b.idx[n.split("=")[1]] if not n.startswith("sound.form") else -1
                             for n in NODES])

    def field(self, a, D=3, gamma=0.5, x1=None):
        x = np.zeros(self.b.N, np.float32) if x1 is None else x1
        if self.inject:
            m = (self.inj >= 0) & (a > 0)
            np.add.at(x, self.inj[m], a[m])
        s, y = x.copy(), x
        for d in range(1, D):
            y = np.asarray(self._P @ y, dtype=np.float32)
            s += gamma ** d * y
        return s

    def scores(self, act, D=3, gamma=0.5):
        a = self.vec(act)
        x = np.zeros(self.b.N, np.float32)
        x[self.widx] = a @ self.V
        s = self.field(a, D, gamma, x)
        return np.log(s[:self.F.n].astype(np.float64) + 1e-8), a, s


def probe(H, glyphs, feats):
    """innate lattice only (no learned paths): hot-minus-calm energy per sound."""
    hot = [H.b.idx[w] for w in HOT]; calm = [H.b.idx[w] for w in CALM]
    out = []
    for g in glyphs:
        s = H.field(H.vec(act(g)))
        out.append(np.log(s[hot] + 1e-8).mean() - np.log(s[calm] + 1e-8).mean())
    out = np.array(out)
    F = np.array(feats)
    aggressive = np.array([g["sound.energy.loudness"] == "loud" and g["sound.timbre.roughness"] == "harsh"
                           for g in glyphs])
    gentle = np.array([g["sound.energy.loudness"] == "quiet" and g["sound.timbre.roughness"] == "smooth"
                       for g in glyphs])
    corr = {r: float(np.corrcoef(F[:, k], out)[0, 1]) for k, (_, r, _) in enumerate(ROLES)}
    return {"loud+harsh sounds: hot-calm": float(out[aggressive].mean()), "n_loud_harsh": int(aggressive.sum()),
            "quiet+smooth sounds: hot-calm": float(out[gentle].mean()), "n_quiet_smooth": int(gentle.sum()),
            "correlation of hot-calm with each raw feature": corr}


def lights(H, g, k=10):
    s = H.field(H.vec(act(g)))[:H.F.n]
    vals = set(g.values())
    order = [i for i in np.argsort(-s)[:k + 20] if H.b.words[i] not in vals][:k]
    return [H.b.words[i] for i in order]


if __name__ == "__main__":
    t0 = time.time()
    z = np.load(HERE / "data" / "dev" / "speech.npz")
    vocab = [w for _, ws in S.STAGES for w in ws]
    rng = np.random.default_rng(0)                            # same split as sound.py
    tr_i, te_i, train, test = {}, {}, {}, {}
    for w in vocab:
        spk = z[f"spk_{w}"]
        uniq = sorted(set(spk))
        held = set(rng.choice(uniq, size=max(1, len(uniq) // 5), replace=False))
        tr_i[w] = [i for i, s in enumerate(spk) if s not in held]
        te_i[w] = [i for i, s in enumerate(spk) if s in held]
        train[w] = [S.ear(z[w][i]) for i in tr_i[w]]
        test[w] = [S.ear(z[w][i]) for i in te_i[w]]
    X = np.concatenate([f for fs in train.values() for f in fs])
    C = S.kmeans(X[rng.choice(len(X), min(60000, len(X)), replace=False)], S.K)   # same atoms
    tr_f = {w: [features(z[w][i]) for i in tr_i[w]] for w in vocab}
    te_f = {w: [features(z[w][i]) for i in te_i[w]] for w in vocab}
    G = Glypher(np.array([f for fs in tr_f.values() for f in fs]))
    tr_g = {w: [G.glyph(f, form(fr, C)) for f, fr in zip(tr_f[w], train[w])] for w in vocab}
    te_g = {w: [G.glyph(f, form(fr, C)) for f, fr in zip(te_f[w], test[w])] for w in vocab}
    print(f"sound glyphs: {sum(map(len, tr_g.values()))} training, {sum(map(len, te_g.values()))} "
          f"held-out-speaker; 15 roles each; {len(NODES)} typed values", flush=True)
    for w in ["no", "happy"]:
        print(f"  '{w}' heard as: " + ", ".join(f"{p.split('.')[-1]}={v}" for p, v in tr_g[w][0].items()))

    base = BB.load_base()
    results = {"cuts": {r: [round(float(c), 3) for c in cut] for (_, r, _), cut in zip(ROLES, G.cuts)}}
    for name, inject, control in [("glyph", True, False), ("glyph CONTROL", True, True)]:
        brain = BB.BigFlow(base)
        H = GlyphHearing(brain, vocab, inject); H.ready()
        if not control:
            allg = [g for gs in te_g.values() for g in gs]
            allf = [f for fs in te_f.values() for f in fs]
            results["emotion probe (innate lattice, untrained)"] = probe(H, allg, allf)
            print("probe:", json.dumps(results["emotion probe (innate lattice, untrained)"], indent=1))
            loud = {"sound.energy.loudness": "loud", "sound.energy.attack": "sudden",
                    "sound.timbre.roughness": "harsh"}
            soft = {"sound.energy.loudness": "quiet", "sound.energy.attack": "gentle",
                    "sound.timbre.roughness": "smooth"}
            for lab, want in [("loud/sudden/harsh", loud), ("quiet/gentle/smooth", soft)]:
                ex = next((g for g in allg if all(g[k] == v for k, v in want.items())), None)
                if ex:
                    li = lights(H, ex)
                    results.setdefault("lights", {})[lab] = li
                    print(f"  a {lab} sound lights up: {', '.join(li)}")
        heard, log = [], []
        for stage, words in S.STAGES:
            heard += words
            pairs = [(act(g), w) for w in words for g in tr_g[w]]
            if control:
                labels = [w for _, w in pairs]; rng.shuffle(labels)
                pairs = [(a, l) for (a, _), l in zip(pairs, labels)]
            order = rng.permutation(len(pairs))
            for epoch in range(3):
                for i in order:
                    H.learn(*pairs[i])
            r = S.evaluate(H, {w: [act(g) for g in te_g[w]] for w in heard}, heard)
            log.append({"stage": stage, "words": len(heard), **r})
            print(f"{name} {stage}: {len(heard)} words | held-out speakers: correct among taught "
                  f"{r['among_taught']:.1%} (chance {r['chance_taught']:.1%}), top-1 among all 68,298 "
                  f"{r['top1_all_68k']:.1%}, median rank {r['median_rank_all']:.0f}", flush=True)
        results[name] = log
        if not control:
            flat = np.argsort(-H.V.ravel())[:12]
            sp_ = [(NODES[s].replace("sound.", ""), H.words[w], round(float(H.V[s, w]), 2))
                   for s, w in (divmod(int(i), len(H.words)) for i in flat)]
            results["strongest paths"] = sp_
            print("strongest sound-glyph paths: " + ", ".join(f"{n} -> {w} {v}" for n, w, v in sp_))
    (HERE / "sound_glyph_results.json").write_text(json.dumps(results, indent=1))
    print(f"done in {time.time() - t0:.0f}s")
