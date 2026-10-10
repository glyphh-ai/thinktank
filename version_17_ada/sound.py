"""
SOUND — ears for the same brain: waveform -> sound primitives -> paths -> word
=============================================================================
Sound is not a separate model. It is a new region of the SAME brain (the
lattice + learned pathways), learned with the SAME rule:

  ear         waveform -> log-mel frames (25 ms window, 10 ms hop, 40 bands),
              each utterance normalized (adaptation)
  primitives  learned WITHOUT labels from heard speech (perceptual tuning, like
              an infant's first year): 64 sound atoms (k-means of frames) and
              the transitions between them (syllable-like primitives).
              No phoneme list, no text.
  paths       sound primitive -> word glyph, formed on co-activation (a word is
              heard while its meaning is active) and learned by the exact local
              rule: activity of the sound primitive x error flowing back to the
              word, through the lattice (thinking continues from the word into
              its meaning: x_1 = sound paths, x_2.. = lattice flow)
  order       babble-level primitives first (unsupervised), then one-syllable
              words, then two-syllable words — simplest first

Test: held-out SPEAKERS only (never heard). The correct word must win among
the taught words and among all 68,298 words. Control: the same training with
labels shuffled across clips (sound no longer tells the word).

Data (not committed): data/dev/speech.npz — Google Speech Commands v0.02
(CC-BY 4.0), 400 clips per word.

Run:  python sound.py
"""

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import big_brain as BB

HERE = Path(__file__).parent
STAGES = [("one syllable", ["no", "go", "up", "yes", "on", "off", "two", "wow",
                            "dog", "cat", "bed", "bird", "tree", "down", "house"]),
          ("two syllables", ["happy", "seven", "zero"])]
K = 64                                                        # sound atoms


# ============================================================================
# the ear
# ============================================================================
def mel_filters(sr=16000, n_fft=400, n_mels=40):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = imel(np.linspace(mel(60), mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


FB = mel_filters()


def ear(x):
    """waveform (int16, 1 s) -> frames (T, 40) of normalized log-mel energy, voiced part only."""
    x = x.astype(np.float32) / 32768
    frames = np.lib.stride_tricks.sliding_window_view(x, 400)[::160] * np.hanning(400)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    logmel = np.log(spec @ FB.T + 1e-8)
    energy = logmel.max(1)
    keep = energy > energy.max() - 6                          # the sounding part
    f = logmel[keep]
    return (f - f.mean(0)) / (f.std(0) + 1e-6)


# ============================================================================
# primitives (unsupervised)
# ============================================================================
def kmeans(X, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None]
        a = d.argmin(1)
        for j in range(k):
            if (a == j).any():
                C[j] = X[a == j].mean(0)
    return C


def primitives(frames, C):
    """frames -> sound-node activity: atoms + transitions between distinct atoms."""
    d = (frames ** 2).sum(1)[:, None] - 2 * frames @ C.T + (C ** 2).sum(1)[None]
    seq = d.argmin(1)
    collapsed = [int(s) for i, s in enumerate(seq) if i == 0 or s != seq[i - 1]]
    act = {}
    for s in collapsed:
        act[s] = act.get(s, 0) + 1.0
    for a, b in zip(collapsed, collapsed[1:]):
        act[K + a * K + b] = act.get(K + a * K + b, 0) + 1.0
    tot = sum(act.values())
    return {k: v / tot for k, v in act.items()}, collapsed


# ============================================================================
# the sound region in the same brain
# ============================================================================
class Hearing:
    def __init__(self, brain, words):
        self.F = brain                                        # the lattice brain
        self.b = brain.b
        self.n_sound = K + K * K
        self.words = list(words)
        self.widx = np.array([self.b.idx[w] for w in self.words])
        self.V = np.zeros((self.n_sound, len(self.words)), np.float32)   # paths sound -> word
        self.formed = np.zeros_like(self.V, dtype=bool)                  # formed on co-activation

    def vec(self, act):
        v = np.zeros(self.n_sound, np.float32)
        for k, a in act.items():
            v[k] = a
        return v

    def scores(self, act, D=3, gamma=0.5):
        a = self.vec(act)
        x = np.zeros(self.b.N, np.float32)
        x[self.widx] = a @ self.V                              # x1: sound paths into words
        s, y = x.copy(), x
        for d in range(1, D):                                 # x2..: thinking through the lattice
            y = np.asarray(self._P @ y, dtype=np.float32)
            s += gamma ** d * y
        return np.log(s[:self.F.n].astype(np.float64) + 1e-8), a, s

    def learn(self, act, word, eta=0.05, D=3, gamma=0.5):
        """Exact local rule: dL/dV[s, w] = activity(s) x backward error(w)."""
        j = self.words.index(word)
        sc, a, s = self.scores(act, D, gamma)
        self.formed[a > 0, j] = True                          # paths form on co-activation
        p = np.exp(sc - sc.max()); p /= p.sum()
        lam = np.zeros(self.b.N, np.float32)
        lam[:self.F.n] = (p - (np.arange(self.F.n) == self.widx[j])) / (s[:self.F.n] + 1e-8)
        back, y = lam.copy(), lam
        for d in range(1, D):                                 # Σ_d γ^d (Pᵀ)^d λ
            y = np.asarray(self._PT @ y, dtype=np.float32)
            back += gamma ** d * y
        g = np.outer(a, back[self.widx]) * self.formed
        g /= np.abs(g).max() + 1e-30                          # normalized step
        self.V = np.maximum(0.0, self.V - eta * g)
        return -np.log(p[self.widx[j]] + 1e-300)

    def ready(self):
        self._P = self.F.P()
        self._PT = self._P.T.tocsr()


def evaluate(H, data, words):
    taught = [H.b.idx[w] for w in words]
    hit_t = hit_all = n = 0
    ranks = []
    for w, acts in data.items():
        for act in acts:
            sc, _, _ = H.scores(act)
            n += 1
            sub = sc[taught]
            hit_t += words[int(np.argmax(sub))] == w
            r = (sc > sc[H.b.idx[w]]).sum()
            ranks.append(r); hit_all += r == 0
    return {"among_taught": hit_t / n, "chance_taught": 1 / len(words),
            "top1_all_68k": hit_all / n, "median_rank_all": float(np.median(ranks) + 1)}


if __name__ == "__main__":
    t0 = time.time()
    z = np.load(HERE / "data" / "dev" / "speech.npz")
    vocab = [w for _, ws in STAGES for w in ws]
    rng = np.random.default_rng(0)
    train, test = {}, {}
    for w in vocab:
        spk = z[f"spk_{w}"]
        uniq = sorted(set(spk))
        held = set(rng.choice(uniq, size=max(1, len(uniq) // 5), replace=False))
        tr = [i for i, s in enumerate(spk) if s not in held]
        te = [i for i, s in enumerate(spk) if s in held]
        train[w] = [ear(z[w][i]) for i in tr]
        test[w] = [ear(z[w][i]) for i in te]
    print(f"ears: {sum(map(len, train.values()))} training clips, "
          f"{sum(map(len, test.values()))} test clips from held-out speakers")

    # perceptual tuning: primitives from heard speech, no labels
    X = np.concatenate([f for fs in train.values() for f in fs])
    C = kmeans(X[rng.choice(len(X), min(60000, len(X)), replace=False)], K)
    tr_act = {w: [primitives(f, C)[0] for f in fs] for w, fs in train.items()}
    te_act = {w: [primitives(f, C)[0] for f in fs] for w, fs in test.items()}
    ex = primitives(train["no"][0], C)[1]
    print(f"primitives: {K} atoms learned from {len(X):,} frames; 'no' sounds like atoms {ex}")

    base = BB.load_base()
    results = {}
    for control in (False, True):
        brain = BB.BigFlow(base)                              # the same lattice brain
        H = Hearing(brain, vocab); H.ready()
        heard = []
        log = []
        for stage, words in STAGES:
            heard += words
            pairs = [(a, w) for w in words for a in tr_act[w]]
            if control:                                       # labels shuffled across clips
                labels = [w for _, w in pairs]; rng.shuffle(labels)
                pairs = [(a, l) for (a, _), l in zip(pairs, labels)]
            order = rng.permutation(len(pairs))
            for epoch in range(3):
                for i in order:
                    H.learn(*pairs[i])
            r = evaluate(H, {w: te_act[w] for w in heard}, heard)
            log.append({"stage": stage, "words": len(heard), **r})
            print(f"{'CONTROL ' if control else ''}{stage}: {len(heard)} words heard | "
                  f"held-out speakers: correct among taught {r['among_taught']:.1%} "
                  f"(chance {r['chance_taught']:.1%}), top-1 among all 68,298 "
                  f"{r['top1_all_68k']:.1%}, median rank {r['median_rank_all']:.0f}", flush=True)
        results["control" if control else "real"] = log
        if not control:
            flat = np.argsort(-H.V.ravel())[:12]
            print("strongest sound paths: " + ", ".join(
                f"{'atom' if s < K else 'trans'}{s if s < K else (s - K) // K}"
                f"{'' if s < K else '>' + str((s - K) % K)} -> {H.words[w]} {H.V[s, w]:.2f}"
                for s, w in (divmod(int(i), len(H.words)) for i in flat)))
    (HERE / "sound_results.json").write_text(json.dumps(results, indent=1))
    print(f"done in {time.time() - t0:.0f}s")
