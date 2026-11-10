"""step14_corrections.py — a correction replaces the nearest fact about the same thing.

step13: through ingest a correction lands only 70% (the old value keeps
30%). The update is rewritten well ("Piheba Fupu's account manager is
Bada."), but learn() only treats it as a correction when the context is
word for word one it has seen; the first note's rewrite often differs
slightly ("...account manager on file is Puroni."), so the memory keeps
both as two facts.

Here learn() also looks for a NEAR context: when a new fact's context has
no exact match but sits closer to a stored one than two different facts
ever sit, and that stored one continues differently, the stored one is the
old version of this fact: its count is halved (as an exact re-telling does)
before the new one is stored.

  how close is "closer than two different facts ever sit"
      tau_c = the largest similarity between two contexts with different
      continuations in a dev store of clean, distinct facts (seed 1, the
      same four facts per person, so "X's account manager is" against
      "X's home town is" is in there). Set before any test fact is seen.

Same data as step13 (300 "far" notes, the 14B's v2 rewrite, a third of the
cases updated). Two memories: step13's (exact re-telling only) and this
one. Asked: every fact before the updates (does the near rule eat distinct
facts while teaching?), then the corrected manager (new / old), untouched
cases' managers, and the corrected cases' OTHER facts (town, product, code:
the nearest neighbours of the fact that changed).

Run:  QWEN=Qwen3-14B-4bit python step14_corrections.py
"""

import argparse
import json
import os
import re
import time

import numpy as np

import qwen
import step3_glyphh as s3
import step13_ingest as s13
from step1_adapter import Arm, HERE, book, recall
from step12_unstructured import ATTRS, NEAR, UPDATE, groups, make_cases, note


class Replacing(s3.GlyphhMemory):
    def __init__(self, tok, tau_c, **kw):
        super().__init__(tok, **kw)
        self.tau_c, self.replaced = tau_c, 0

    def learn(self, tokens):
        ctx = self.contexts(tokens)
        for t in range(len(tokens) - 1):
            if not ctx[t].any():
                continue
            nxt = tokens[t + 1]
            if ctx[t].tobytes() not in self._index and len(self.counts):
                s = self.keys @ ctx[t]
                j = int(s.argmax())
                c = self.counts[j]
                if s[j] >= self.tau_c and c and nxt not in c:
                    for other in list(c):
                        c[other] //= 2
                        if not c[other]:
                            del c[other]
                    self.replaced += 1
        super().learn(tokens)


class Settled(s3.GlyphhMemory):
    """step14c. The nearest-key rule (above) could not work: "X's account manager is" and "... is now" sit at
    0.91, further apart than two different facts of one person (0.95), because the word being written ("is",
    "now") is always in the key. A correction is judged on SETTLED contexts instead: at the end of a finished
    word, with stop words out (so "is" and "is now" are the same context), and only where a content word comes
    next (a value, not "is"). A stored context of that kind at tau_c or closer that continues differently is
    the old version: halved."""

    def __init__(self, tok, tau_c, **kw):
        super().__init__(tok, **kw)
        self.tau_c, self.replaced = tau_c, 0
        self._set = np.zeros((1024, s3.DIM), np.float32)
        self._flag = np.zeros(1024, bool)

    def settled(self, tokens):
        pieces = [self.tok.decode([x]) for x in tokens]
        out, flag, text = np.zeros((len(tokens), s3.DIM), np.float32), np.zeros(len(tokens), bool), ""
        for t, p in enumerate(pieces):
            text = (text + p)[-400:]
            nxt = pieces[t + 1] if t + 1 < len(pieces) else " "
            if nxt[:1].isalpha():                              # mid-word: nothing settled here
                continue
            word = re.match(r"\s*([A-Za-z]+)", "".join(pieces[t + 1:t + 8]))
            flag[t] = bool(word) and word.group(1).lower() not in s3.STOP
            out[t] = self.context_of(text + " ")               # the trailing space: no word is "being written"
        return out, flag

    def learn(self, tokens):
        ctx = self.contexts(tokens)
        sc, flag = self.settled(tokens)
        n0 = len(self.counts)
        for t in range(len(tokens) - 1):
            if not n0 or not flag[t] or not sc[t].any():
                continue
            s = self._set[:n0] @ sc[t]
            for j in np.flatnonzero((s >= self.tau_c) & self._flag[:n0]):
                c = self.counts[j]
                if c and tokens[t + 1] not in c:
                    for other in list(c):
                        c[other] //= 2
                        if not c[other]:
                            del c[other]
                    self.replaced += 1
        super().learn(tokens)
        if len(self._set) < len(self._buf):
            grow = len(self._buf) - len(self._set)
            self._set = np.vstack([self._set, np.zeros((grow, s3.DIM), np.float32)])
            self._flag = np.concatenate([self._flag, np.zeros(grow, bool)])
        for t in range(len(tokens) - 1):
            if ctx[t].any():
                j = self._index[ctx[t].tobytes()]
                if j >= n0:
                    self._set[j], self._flag[j] = sc[t], flag[t]


class SettledKeys(Settled):
    """step14d. Replacing stored contexts (14b, 14c) changed nothing, because the trouble is in the KEY, not in
    what is replaced: the 30% that fail are the 32 updates worded "is now". Their value is stored under
    "... manager now" and the question ends "... manager is", so the new value's key is further from the
    question than the (emptied) old one and never gets mixed in. The word being written was kept in the key
    even when it is a finished stop word, to protect a name's first piece ("He" of "Heliro"). But when the
    NEXT token starts a new word, the word is finished: so there the key drops it if it is a stop word. "is",
    "is now" and the question then share one key, and an update is an exact re-telling again."""

    def contexts(self, tokens):
        pieces = [self.tok.decode([x]) for x in tokens]
        out, text = np.zeros((len(tokens), s3.DIM), np.float32), ""
        for t, p in enumerate(pieces):
            text = (text + p)[-400:]
            new_word = t + 1 < len(pieces) and not pieces[t + 1][:1].isalpha()
            out[t] = self.context_of(text + " " if new_word else text)
        return out


def settled_gap(q, n, delta, cls=None):
    """tau_c for Settled: the closest two flagged contexts of different dev facts get, identical ones aside
    (an identical context is the same context: the exact rule's case)."""
    dev = (cls or Settled)(q.tok, 2.0, delta=delta)
    fact, fid = [], 0
    for c in make_cases(n, seed=1):
        for a in ATTRS:
            dev.learn(q.encode(NEAR[a][0].format(p=c["p"], v=c[a])))
            fact += [fid] * (len(dev.counts) - len(fact))
            fid += 1
    rows = np.flatnonzero(dev._flag[:len(dev.counts)])
    K, fact = dev._set[rows], np.array(fact)[rows]
    top = np.array([max(dev.counts[j], key=dev.counts[j].get) if dev.counts[j] else -1 for j in rows])
    worst = 0.0
    for i in range(0, len(K), 2048):
        S = K[i:i + 2048] @ K.T
        S[top[i:i + 2048, None] == top[None, :]] = -1
        S[fact[i:i + 2048, None] == fact[None, :]] = -1
        S[S >= 0.999] = -1
        worst = max(worst, float(S.max()))
    _, s = dev.nearest(dev.contexts(q.encode(book("frankenstein", 84)[50000:58000])))
    dev.tau_fire = float(np.percentile(s, 99.5))
    return worst if cls is None else (worst, dev.tau_fire)


def distinct_gap(q, n, delta):
    """tau_c and tau from a dev store of clean distinct facts."""
    dev = s3.GlyphhMemory(q.tok, delta=delta)
    fact, fid = [], 0                                           # which fact each key came from
    for c in make_cases(n, seed=1):
        for a in ATTRS:
            dev.learn(q.encode(NEAR[a][0].format(p=c["p"], v=c[a])))
            fact += [fid] * (len(dev.counts) - len(fact))
            fid += 1
    K = dev.keys
    fact = np.array(fact)
    top = np.array([max(c, key=c.get) for c in dev.counts])
    worst = 0.0
    for i in range(0, len(K), 2048):
        S = K[i:i + 2048] @ K.T
        S[top[i:i + 2048, None] == top[None, :]] = -1          # the same continuation (and itself)
        # step14b: nor two spots of one fact ("...is pur" / "...is puro" sit at 0.97 and set tau_c so high
        # that nothing was ever replaced)
        S[fact[i:i + 2048, None] == fact[None, :]] = -1
        worst = max(worst, float(S.max()))
    _, s = dev.nearest(dev.contexts(q.encode(book("frankenstein", 84)[50000:58000])))
    return worst, float(np.percentile(s, 99.5))


def main(n, sample, out, delta=0.05, only=None):
    q = qwen.Qwen()
    s13.V2 = True
    cases = make_cases(n)
    rng = np.random.default_rng(11)
    texts = [note(c, "far", rng) for c in cases]
    third = n // 3
    updates = [UPDATE["far"].format(p=c["p"], v=c["new"]) for c in cases[:third]]
    cache = os.path.join(HERE, "step14_lines.json")
    if os.path.exists(cache):
        kept, up_kept = json.load(open(cache))
    else:
        t = time.time()
        kept, _ = s13.ingest(q, cases, texts)
        up_kept, _ = s13.ingest(q, [{**c, "manager": c["new"]} for c in cases[:third]], updates)
        json.dump([kept, up_kept], open(cache, "w"), indent=1)
        print(f"rewrote {n} notes and {third} updates in {time.time() - t:.0f}s", flush=True)
    tau_c, tau = distinct_gap(q, n, delta)
    print(f"{qwen.NAME}; {n} notes, {third} updated; tau {tau:.2f}; two different dev facts never sit closer than "
          f"{tau_c:.3f} (tau_c)\n", flush=True)

    pick = np.random.default_rng(3).choice(n, min(sample, n), replace=False)
    fixed = np.random.default_rng(4).choice(third, min(sample, third), replace=False)
    rest = third + np.random.default_rng(5).choice(n - third, min(sample, n - third), replace=False)
    results = {"tau": tau, "tau_c": tau_c}
    for name in ("exact", "nearest", "settled", "settledkeys"):
        if only and name not in only:
            continue
        t = time.time()
        if name == "settledkeys":
            tau_s, tau_k = settled_gap(q, n, delta, SettledKeys)
            tau_s += 1e-3
            print(f"settled keys: tau {tau_k:.2f}; two different dev facts never sit closer than {tau_s:.3f}", flush=True)
            mem = SettledKeys(q.tok, tau_s, delta=delta)
            results["tau_settledkeys"] = [tau_k, tau_s]
        elif name == "settled":
            tau_s = settled_gap(q, n, delta) + 1e-3
            print(f"settled: two different dev facts' settled contexts never sit closer than {tau_s:.3f}", flush=True)
            mem = Settled(q.tok, tau_s, delta=delta)
            results["tau_settled"] = tau_s
        else:
            mem = Replacing(q.tok, tau_c if name == "nearest" else 2.0, delta=delta)
        mem.tau = tau_k if name == "settledkeys" else tau
        arm = Arm(q, name, exact=mem)
        arm.teach([x for k in kept for x in k])
        row = {"arm": name, "replaced_teaching": mem.replaced, **s13.ask_all(arm, cases, pick)}
        mem.replaced = 0
        arm.teach([x for k in up_kept for x in k])
        row["replaced_updates"] = mem.replaced
        r2 = np.random.default_rng(23)
        for form, tag in enumerate(("qa", "reworded")):
            nw, old = recall(arm.answer_bits(groups(cases, fixed, "manager", form, r2, "new", "manager")), True)
            row[f"new_{tag}"], row[f"old_{tag}"] = nw, old
            row[f"untouched_{tag}"], _ = recall(arm.answer_bits(groups(cases, rest, "manager", form, r2)))
        for a in ("town", "product", "code"):
            row[f"sibling_{a}"], _ = recall(arm.answer_bits(groups(cases, fixed, a, 0, r2)))
        print(f"{name:<8} before updates: recall {row['recall']:4.0%} (contexts replaced while teaching: "
              f"{row['replaced_teaching']}) | {third} updates replaced {row['replaced_updates']} contexts | corrected "
              f"manager: new {row['new_qa']:.0%}/{row['new_reworded']:.0%} old {row['old_qa']:.0%}/"
              f"{row['old_reworded']:.0%} | untouched managers {row['untouched_qa']:.0%}/{row['untouched_reworded']:.0%}"
              f" | the corrected cases' other facts: town {row['sibling_town']:.0%} product "
              f"{row['sibling_product']:.0%} code {row['sibling_code']:.0%}  ({time.time() - t:.0f}s)", flush=True)
        results[name] = row
        with open(os.path.join(HERE, out), "w") as fh:
            json.dump(results, fh, indent=1, default=float)
        del mem, arm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", type=int, default=300)
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--out", default="step14_results.json")
    ap.add_argument("--arms", default="")
    a = ap.parse_args()
    main(a.notes, a.sample, a.out, only=a.arms.split(",") if a.arms else None)
