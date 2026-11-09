"""step3_glyphh.py — the counter memory keyed on a glyphh instead of tokens.

step1's exact memory keys each next-token count on a hash of the last 4/8/12
Qwen tokens, so it only fires when the question is worded the way the fact
was told (99% recall, 6% paraphrase). Here the key is a glyphh of what the
context is about, built with Ada's own reading (glyphh-ai ada-private):

  words   the last WINDOW content words, each its lexicon glyph (WordNet +
          Moby: the word tilted toward its synonyms, near words and plain
          form, so "named" reads with "name"); a word the lexicon doesn't
          know (an invented name) is its own symbol, at NAME_VOICE
  primes  the NSM primes the window relates to (glyphh_nsm.decompose), at
          PRIME_VOICE

  learn   at each told position, the context's glyphh -> counts of the next
          token; a glyphh already stored (the identical context) updates its counts the
          way exact does (rivals halved), so a correction replaces
  refine  at each position, the nearest stored glyphh; above tau its counts
          mix into Qwen's next-token probability, as exact's do

tau is set on dev facts (seed 1) against unrelated text, never on the test
facts. Same 300 facts, same phases, same scoring as step1.

Run:  python step3_glyphh.py [--facts 300] [--passes 2] [--arms glyphh,exact]
"""

import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np

import memory
import qwen
from step1_adapter import Arm, HERE, book, make_facts, run_arm, sentence

sys.path.insert(0, os.path.expanduser("~/development/glyphh/ada-private"))
from glyphh_kernel import symbol  # noqa: E402
from glyphh_lexicon import LEXICON  # noqa: E402
from glyphh_nsm import decompose  # noqa: E402

DIM, SEED = 2048, 7
WINDOW = 8
NAME_VOICE, WORD_VOICE, PRIME_VOICE = 2.0, 1.0, 0.5
SAME = 0.999
STOP = set("""a an the of to and or is are was were be been it its it's this that
these those in on at for with by from as q what who which""".split())
_WORD = re.compile(r"[a-z]+")


class GlyphhMemory:
    def __init__(self, tok, tau=0.5, beta=1.0, delta=0.0):
        # delta 0: the single nearest key (step3). delta > 0: every key above
        # tau within delta of the best, its counts weighted by similarity
        # (step4b), so a wording whose nearest key sits one token early
        # ("...keeps a cat" -> " named") still gets the next key's answer
        self.tok, self.tau, self.beta, self.delta = tok, tau, beta, delta
        self._buf = np.zeros((1024, DIM), np.float32)
        self.counts = []
        self._index = {}             # exact glyphh bytes -> row (a re-told context)
        self._word = {}

    @property
    def keys(self):
        return self._buf[:len(self.counts)]

    # -------------------------------------------------------------- the key
    def _wglyph(self, w):
        g = self._word.get(w)
        if g is None:
            known = LEXICON.entry(w) is not None or LEXICON.lemma(w) is not None
            g = LEXICON.glyph(w, SEED, DIM, reach=2).astype(np.float32) * (WORD_VOICE if known else NAME_VOICE)
            self._word[w] = g
        return g

    def contexts(self, tokens):
        """the glyphh of the context ending at each position (row t keys tokens[t + 1])."""
        pieces = [self.tok.decode([x]) for x in tokens]
        out = np.zeros((len(tokens), DIM), np.float32)
        text = ""
        for t, p in enumerate(pieces):
            text = (text + p)[-400:]
            words = [w for w in _WORD.findall(text.lower()) if w not in STOP][-WINDOW:]
            if not words:
                continue
            v = sum(self._wglyph(w) for w in words)
            for prime in decompose(" ".join(words)):
                v = v + symbol(f"nsm:{prime}", SEED, DIM).astype(np.float32) * PRIME_VOICE
            out[t] = v / (np.linalg.norm(v) or 1.0)
        return out

    def nearest(self, ctx):
        if not len(self.keys):
            return np.full(len(ctx), -1), np.zeros(len(ctx))
        s = ctx @ self.keys.T
        j = s.argmax(1)
        return j, s[np.arange(len(ctx)), j]

    # -------------------------------------------------------- learn / refine
    def learn(self, tokens):
        ctx = self.contexts(tokens)
        for t in range(len(tokens) - 1):
            if not ctx[t].any():
                continue
            nxt = tokens[t + 1]
            h = ctx[t].tobytes()
            j = self._index.get(h)
            if j is not None:
                c = self.counts[j]
                for other in [o for o in c if o != nxt]:
                    c[other] //= 2
                    if not c[other]:
                        del c[other]
                c[nxt] = c.get(nxt, 0) + 1
            else:
                if len(self.counts) == len(self._buf):
                    self._buf = np.vstack([self._buf, np.zeros_like(self._buf)])
                self._buf[len(self.counts)] = ctx[t]
                self._index[h] = len(self.counts)
                self.counts.append({nxt: 1})

    def refine(self, tokens, lp, start=0):
        lp = lp.copy()
        ctx = self.contexts(tokens)
        if self.delta > 0:
            return self._refine_mix(tokens, lp, ctx, start)
        js, ss = self.nearest(ctx[start:len(tokens) - 1])
        for t, j, s in zip(range(start, len(tokens) - 1), js, ss):
            if j < 0 or s < self.tau:
                continue
            c = self.counts[j]
            p = np.exp(lp[t])
            p = (c.get(tokens[t + 1], 0) + self.beta * p) / (sum(c.values()) + self.beta)
            lp[t] = np.log(max(p, 1e-30))
        return lp

    def _refine_mix(self, tokens, lp, ctx, start):
        if not len(self.counts) or start >= len(tokens) - 1:
            return lp
        S = ctx[start:len(tokens) - 1] @ self.keys.T
        for row, t in zip(S, range(start, len(tokens) - 1)):
            best = row.max()
            if best < self.tau:
                continue
            hit = np.flatnonzero(row >= max(self.tau, best - self.delta))
            num = den = 0.0
            for j in hit:
                c, w = self.counts[j], float(row[j])
                num += w * c.get(tokens[t + 1], 0)
                den += w * sum(c.values())
            p = (num + self.beta * np.exp(lp[t])) / (den + self.beta)
            lp[t] = np.log(max(p, 1e-30))
        return lp


def calibrate(q, pct=99.5):
    """tau = the pct percentile of nearest-key similarity from unrelated text
    (Frankenstein) to a memory of dev facts (seed 1), so it rarely fires there."""
    dev = make_facts(100, seed=1)
    m = GlyphhMemory(q.tok)
    for f in dev:
        m.learn(q.encode(sentence(f, "answer")))
    other = q.encode(book("frankenstein", 84)[50000:58000])
    _, s = m.nearest(m.contexts(other))
    tau = float(np.percentile(s, pct))
    # what the dev facts get at the answer position, worded both ways
    hit = []
    for key in ("taught", "para"):
        sims = []
        for f in dev:
            toks = q.encode(f[key] + f["answer"])
            a = len(q.encode(f[key]))
            sims.append(m.nearest(m.contexts(toks)[a - 1:a])[1][0])
        hit.append(np.mean(np.array(sims) > tau))
    print(f"  unrelated text {pct}th pct similarity {tau:.3f}; dev answer keys above it: "
          f"taught {hit[0]:.0%}, paraphrase {hit[1]:.0%}", flush=True)
    return tau


def main(n, passes, arms, out):
    pp = book("pride_and_prejudice", 1342)[:20000]
    q = qwen.Qwen()
    print(f"{qwen.NAME}, {n} invented facts, glyphh dim {DIM}, window {WINDOW} words\n")
    rows = []
    if "glyphh" in arms:
        print("calibrating tau on dev facts")
        tau = calibrate(q)
        print()
    for a in arms:
        if a == "glyphh":
            arm = Arm(q, "qwen + glyphh memory", exact=GlyphhMemory(q.tok, tau))
        elif a == "exact":
            arm = Arm(q, "qwen + exact memory", exact=memory.ExactMemory())
        else:
            continue
        rows += run_arm(arm, make_facts(n), pp, passes)
        print()
        with open(os.path.join(HERE, out), "w") as fh:
            json.dump(rows, fh, indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", type=int, default=300)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--arms", default="glyphh")
    ap.add_argument("--out", default="step3_results.json")
    a = ap.parse_args()
    main(a.facts, a.passes, a.arms.split(","), a.out)
