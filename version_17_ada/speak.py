"""
SPEAK — a mouth for the same brain: babble, listen to yourself, imitate
======================================================================
Until now the brain could only hear. Here it gets a MOUTH and learns to use it
the way an infant does — not from text, not from a voice synthesizer that
already knows words, but from its own ears:

  mouth     a source-filter vocal tract with 9 articulators, each 0..1:
              voice (vocal folds on), pitch, jaw (open), tongue front/back,
              lips (closed), tip (tongue tip closure, d/n/t), dorsum (back
              closure, g/k), velum (nose open, m/n), hiss (frication, s/f/h)
            a motor program is 6 keyframes of the 9 articulators (~0.5 s),
            smoothly interpolated. The mouth knows nothing about speech.
  babble    random motor programs; it hears each one through the SAME ear and
            the SAME sound primitives learned from real speech (sound.py), and
            remembers what each movement sounded like.
  imitate   a caregiver says a word (real recordings, Speech Commands). The
            brain recalls the babble that sounded most like it (memory), then
            refines it by reinforcement (the three-factor rule): try a slightly
            different movement, listen, and move toward movements whose sound
            the brain's own hearing recognizes as the word it meant to say.
            The learned program becomes the word's motor path.
  test      can it SAY the word consistently? 20 productions with natural motor
            noise each, judged by
              (a) its own hearing (the brain trained in sound.py),
              (b) an INDEPENDENT judge that never touches the brain: softmax
                  regression on log-mel, trained on real speakers only,
              and by you: wav files are written so a human can listen.
            Before learning (recall from babble only) and random babble are the
            baselines.

Run:  python speak.py
"""

import json
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import lfilter

import big_brain as BB
import sound as S

HERE = Path(__file__).parent
OUT = HERE / "speak_results"
SR = 16000
ART = ["voice", "pitch", "jaw", "front", "lips", "tip", "dorsum", "velum", "hiss"]
KEYS = 6
BLOCK = 160                                                    # 10 ms
WORDS = S.STAGES[0][1]                                         # the 15 one-syllable words it has heard


# ============================================================================
# the mouth
# ============================================================================
def resonate(x, f, bw, zi):
    r = np.exp(-np.pi * bw / SR)
    a = [1, -2 * r * np.cos(2 * np.pi * f / SR), r * r]
    b = [1 - r]
    y, zi = lfilter(b, a, x, zi=zi)
    return y, zi


def mouth(prog, rng, dur=0.5):
    """motor program (KEYS x 9, 0..1) -> waveform int16, 1 s."""
    nb = int(dur * SR / BLOCK)
    t = np.linspace(0, KEYS - 1, nb)
    A = np.array([np.interp(t, np.arange(KEYS), prog[:, k]) for k in range(len(ART))]).T
    out, phase = [], 0.0
    zi = {k: np.zeros(2) for k in ("f1", "f2", "f3", "n", "h")}
    for voice, pitch, jaw, front, lips, tip, dorsum, velum, hiss in A:
        f0 = 90 + 260 * pitch
        n = np.arange(BLOCK)
        ph = phase + n * f0 / SR
        src = (np.diff(np.floor(np.concatenate([[phase - f0 / SR], ph]))) > 0).astype(float)
        phase = ph[-1] + f0 / SR
        src = lfilter([1], [1, -0.97], src) * voice                 # glottal pulses
        src += rng.normal(0, 0.02, BLOCK) * (1 - voice)            # breath
        closure = max(lips, tip, dorsum)
        f1 = 220 + 700 * jaw * (1 - 0.7 * closure)
        f2 = 650 + 1600 * front
        f2 = f2 * (1 - tip) + 1800 * tip
        f2 = f2 * (1 - dorsum) + 2100 * dorsum
        f2 -= 350 * lips
        f3 = 2600 - 400 * dorsum + 200 * front
        oral = (1 - closure) ** 2
        y1, zi["f1"] = resonate(src, f1, 80, zi["f1"])
        y2, zi["f2"] = resonate(y1, f2, 120, zi["f2"])
        y3, zi["f3"] = resonate(y2, f3, 200, zi["f3"])
        yn, zi["n"] = resonate(src, 260, 100, zi["n"])
        noise = rng.normal(0, 1, BLOCK) * hiss
        yh, zi["h"] = resonate(noise, 3500 + 2500 * front, 1500, zi["h"])
        out.append(y3 * oral * 40 + yn * velum * 6 + yh * 0.3)
    y = np.concatenate(out)
    y = y / (np.abs(y).max() + 1e-9) * 0.5
    full = rng.normal(0, 1e-4, SR)
    s = (SR - len(y)) // 2
    full[s:s + len(y)] += y
    return (full * 32767).astype(np.int16)


# ============================================================================
# independent judge (never touches the brain)
# ============================================================================
def judge_feat(x):
    f = S.ear(x)
    idx = np.linspace(0, len(f) - 1, 10).round().astype(int)
    return f[idx].ravel()


class Judge:
    def __init__(self, X, y, n, epochs=300, lr=0.5):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.W = np.zeros((Z.shape[1], n)); self.b = np.zeros(n)
        Y = np.eye(n)[y]
        for _ in range(epochs):
            P = self.prob(Z, z=True)
            G = (P - Y) / len(Z)
            self.W -= lr * (Z.T @ G + 1e-3 * self.W); self.b -= lr * G.sum(0)

    def prob(self, X, z=False):
        Z = X if z else (X - self.mu) / self.sd
        L = Z @ self.W + self.b
        P = np.exp(L - L.max(1, keepdims=True))
        return P / P.sum(1, keepdims=True)


# ============================================================================
# the brain's hearing (as sound.py) and imitation
# ============================================================================
def hear_word(H, act):
    sc, _, _ = H.scores(act)
    sub = sc[[H.b.idx[w] for w in WORDS]]
    p = np.exp(sub - sub.max()); p /= p.sum()
    return p


def produce(prog, C, rng):
    x = mouth(prog, rng)
    return x, S.primitives(S.ear(x), C)[0]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def imitate(H, w, template, memory, C, rng, iters=120, pop=8, sigma=0.08, eta=0.6):
    j = WORDS.index(w)
    # recall: the babble that sounded most like what was heard
    sims = [cos(m, template) for m in memory["vec"]]
    prog = memory["prog"][int(np.argmax(sims))].copy()
    recalled = prog.copy()
    base, hist = None, []
    for it in range(iters):
        eps = rng.normal(0, sigma, (pop // 2,) + prog.shape)
        eps = np.concatenate([eps, -eps])                      # antithetic
        R = []
        for e in eps:
            _, a = produce(np.clip(prog + e, 0, 1), C, rng)
            R.append(np.log(hear_word(H, a)[j] + 1e-12))
        R = np.array(R)
        base = R.mean() if base is None else 0.9 * base + 0.1 * R.mean()
        adv = (R - R.mean()) / (R.std() + 1e-9)               # reward minus expectation
        prog = np.clip(prog + eta * sigma * (adv[:, None, None] * eps).mean(0) / sigma, 0, 1)
        hist.append(float(R.mean()))
    return recalled, prog, hist


def say_test(H, J, prog, w, C, rng, n=20, noise=0.03):
    j = WORDS.index(w)
    own = jud = 0
    for _ in range(n):
        x, a = produce(np.clip(prog + rng.normal(0, noise, prog.shape), 0, 1), C, rng)
        own += int(np.argmax(hear_word(H, a)) == j)
        jud += int(np.argmax(J.prob(judge_feat(x)[None])[0]) == j)
    return own / n, jud / n


if __name__ == "__main__":
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    z = np.load(HERE / "data" / "dev" / "speech.npz")
    vocab = [w for _, ws in S.STAGES for w in ws]
    rng = np.random.default_rng(0)                             # same split + atoms as sound.py
    train, test, tr_i, te_i = {}, {}, {}, {}
    for w in vocab:
        spk = z[f"spk_{w}"]
        uniq = sorted(set(spk))
        held = set(rng.choice(uniq, size=max(1, len(uniq) // 5), replace=False))
        tr_i[w] = [i for i, s in enumerate(spk) if s not in held]
        te_i[w] = [i for i, s in enumerate(spk) if s in held]
        train[w] = [S.ear(z[w][i]) for i in tr_i[w]]
        test[w] = [S.ear(z[w][i]) for i in te_i[w]]
    X = np.concatenate([f for fs in train.values() for f in fs])
    C = S.kmeans(X[rng.choice(len(X), min(60000, len(X)), replace=False)], S.K)
    tr_act = {w: [S.primitives(f, C)[0] for f in fs] for w, fs in train.items()}

    # the brain's hearing: taught exactly as in sound.py (one-syllable stage)
    base = BB.load_base()
    H = S.Hearing(BB.BigFlow(base), WORDS); H.ready()
    cache = OUT / "hearing_V.npz"
    if cache.exists():
        d = np.load(cache); H.V, H.formed = d["V"], d["formed"]
    else:
        pairs = [(a, w) for w in WORDS for a in tr_act[w]]
        order = rng.permutation(len(pairs))
        for epoch in range(3):
            for i in order:
                H.learn(*pairs[i])
        np.savez_compressed(cache, V=H.V, formed=H.formed)
    te_act = {w: [S.primitives(f, C)[0] for f in test[w]] for w in WORDS}
    hear_acc = np.mean([np.argmax(hear_word(H, a)) == WORDS.index(w) for w in WORDS for a in te_act[w]])
    print(f"hearing (as sound.py): {hear_acc:.1%} on held-out speakers among {len(WORDS)} words", flush=True)

    # independent judge: real speakers only, never the brain
    Xj = np.array([judge_feat(z[w][i]) for w in WORDS for i in tr_i[w]])
    yj = np.array([k for k, w in enumerate(WORDS) for _ in tr_i[w]])
    J = Judge(Xj, yj, len(WORDS))
    Xt = np.array([judge_feat(z[w][i]) for w in WORDS for i in te_i[w]])
    yt = np.array([k for k, w in enumerate(WORDS) for _ in te_i[w]])
    judge_acc = float((J.prob(Xt).argmax(1) == yt).mean())
    print(f"independent judge: {judge_acc:.1%} on held-out real speakers", flush=True)

    # babble: random movements, heard by its own ear
    n_babble = 3000
    memory = {"prog": [], "vec": []}
    for i in range(n_babble):
        p = rng.random((KEYS, len(ART)))
        x, a = produce(p, C, rng)
        memory["prog"].append(p); memory["vec"].append(H.vec(a))
        if i < 3:
            wavfile.write(OUT / f"babble_{i + 1}.wav", SR, x)
    print(f"babbled {n_babble} random movements and listened to each ({time.time() - t0:.0f}s)", flush=True)

    results = {"hearing_acc": float(hear_acc), "judge_acc": judge_acc, "chance": 1 / len(WORDS), "words": {}}
    for w in WORDS:
        template = np.mean([H.vec(a) for a in tr_act[w]], 0)
        recalled, prog, hist = imitate(H, w, template, memory, C, rng)
        rnd = say_test(H, J, rng.random((KEYS, len(ART))), w, C, rng)
        before = say_test(H, J, recalled, w, C, rng)
        after = say_test(H, J, prog, w, C, rng)
        for k in range(3):
            x, _ = produce(np.clip(prog + rng.normal(0, 0.03, prog.shape), 0, 1), C, rng)
            wavfile.write(OUT / f"say_{w}_{k + 1}.wav", SR, x)
        wavfile.write(OUT / f"heard_{w}.wav", SR, z[w][te_i[w][0]])
        results["words"][w] = {"random": rnd, "recalled_from_babble": before, "after_practice": after,
                               "reward_start": hist[0], "reward_end": hist[-1],
                               "program": np.round(prog, 2).tolist()}
        print(f"say '{w}': own hearing / independent judge recognize it  random {rnd[0]:.0%}/{rnd[1]:.0%}"
              f"  | recalled from babble {before[0]:.0%}/{before[1]:.0%}  | after practice "
              f"{after[0]:.0%}/{after[1]:.0%}   (20 tries each)", flush=True)
        (OUT / "results.json").write_text(json.dumps(results, indent=1))
    W = results["words"]
    for k in ("random", "recalled_from_babble", "after_practice"):
        print(f"MEAN {k}: own hearing {np.mean([W[w][k][0] for w in W]):.1%}, "
              f"independent judge {np.mean([W[w][k][1] for w in W]):.1%} (chance {1 / len(WORDS):.1%})")
    print(f"done in {time.time() - t0:.0f}s")
