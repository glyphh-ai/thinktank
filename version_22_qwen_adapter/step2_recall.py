"""step2_recall.py — angle memory as recall: find the sentence, hand it to Qwen.

step1's angle memory nudges the next token directly, and that only works
when the question is worded the way the fact was told. Here the memory
does what memory is for and Qwen does the reading:

  store   every told sentence, as the sign bits of Qwen's layer-L state at
          each of its tokens (2048 bits a token)
  recall  every token of the question looks for stored tokens within tau
          bits (XOR + popcount); each hit votes for its sentence, weighted
          1 / (sentences that token hits), so names count and "the", "a",
          "named" don't
  answer  the top `top` sentences go in front of the question, newest last,
          and Qwen answers

It's retrieval-augmented generation with a binary, angle-matched index.
The question is whether it carries the paraphrase step1 couldn't, without
the in-context arm's cost of putting every fact in the prompt.

Run:  python step2_recall.py [--facts 300] [--layer 4] [--top 2]
"""

import argparse
import json
import os
import time

import numpy as np

import memory
import qwen
import step1_adapter as s1


class SentenceMemory:
    def __init__(self, layer, mu, tau, top):
        self.layer, self.mu, self.tau, self.top = layer, mu, tau, top
        self.code = np.zeros((0, mu.shape[0] // 64), np.uint64)
        self.sid = np.zeros(0, np.int64)
        self.text = []

    def store(self, text, hidden):
        if text in self.text:                  # told again: already held
            return
        c = memory.codes(hidden, self.mu)
        self.code = np.concatenate([self.code, c])
        self.sid = np.concatenate([self.sid, np.full(len(c), len(self.text))])
        self.text.append(text)

    def recall(self, hidden):
        if not self.text:
            return []
        c = memory.codes(hidden, self.mu)
        d = np.empty((len(c), len(self.code)), np.int32)
        for k in range(0, len(c), 64):
            d[k:k + 64] = np.bitwise_count(c[k:k + 64, None, :] ^ self.code[None]).sum(-1)
        votes = np.zeros(len(self.text))
        for row in d <= self.tau:
            hit = np.unique(self.sid[row])
            if len(hit):
                votes[hit] += 1.0 / len(hit)
        order = np.lexsort((-np.arange(len(votes)), -votes))      # most votes, then newest
        best = [i for i in order[:self.top] if votes[i] > 0]
        return sorted(best)                    # newest last


class RecallArm(s1.Arm):
    def __init__(self, q, mem):
        super().__init__(q, f"qwen + angle recall (layer {mem.layer}, top {mem.top})")
        self.mem = mem
        self.hits = []

    def teach(self, texts):
        seqs = [self.q.encode(t) for t in texts]
        for t, (_, h) in zip(texts, self.q.run(seqs, (self.mem.layer,), batch=48)):
            self.mem.store(t, h[self.mem.layer])

    def answer_bits(self, groups, prefix=None):
        prompts = [p for p, _ in groups]
        out = self.q.run([self.q.encode(p) for p in prompts], (self.mem.layer,), batch=48)
        new_groups = []
        for (p, cands), (_, h) in zip(groups, out):
            got = self.mem.recall(h[self.mem.layer])
            ctx = "".join(self.mem.text[i] + "\n" for i in got)
            new_groups.append((ctx + "\n" + p if ctx else p, cands))
        bits = []
        for g in new_groups:                    # each has its own context
            bits.append(super().answer_bits([g])[0])
        return np.array(bits)


def calibrate(q, layer, pct=0.5):
    other = s1.book("frankenstein", 84)[20000:32000]
    toks = q.encode(other)
    wins = [toks[k:k + 512] for k in range(0, len(toks), 512)]
    H = np.concatenate([h[layer] for _, h in q.run(wins, (layer,), batch=4)])
    mu = H.mean(0)
    dev = s1.make_facts(100, seed=1)
    m = SentenceMemory(layer, mu, 0, 1)
    for f, (_, h) in zip(dev, q.run([q.encode(s1.sentence(f, "answer")) for f in dev], (layer,))):
        m.store(s1.sentence(f, "answer"), h[layer])
    d = np.bitwise_count(memory.codes(H[:2000], mu)[:, None, :] ^ m.code[None]).sum(-1).min(1)
    return mu, int(np.percentile(d, pct))


def main(n, layer, top, passes):
    pp = s1.book("pride_and_prejudice", 1342)[:20000]
    q = qwen.Qwen()
    mu, tau = calibrate(q, layer)
    print(f"{qwen.NAME}, layer {layer}, tau {tau} of {mu.shape[0]} bits (0.5th pct of unrelated "
          f"text vs dev memory), top {top}\n")
    t = time.time()
    rows = s1.run_arm(RecallArm(q, SentenceMemory(layer, mu, tau, top)), s1.make_facts(n), pp,
                      passes)
    print(f"  ({time.time() - t:.0f}s)")
    with open(os.path.join(s1.HERE, f"step2_results_L{layer}_top{top}.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", type=int, default=300)
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--passes", type=int, default=2)
    a = ap.parse_args()
    main(a.facts, a.layer, a.top, a.passes)
