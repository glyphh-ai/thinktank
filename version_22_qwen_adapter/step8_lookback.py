"""step8_lookback.py — predict the reply first, then look back at what the rules changed.

step6/7 nudged the live next token and keyed on Qwen's top-5 there. The
sign-off reached 18%, the ban nothing: the stored situations came from
replies written WITH the rules, so "about to say logistic" was never one of
them. Here the rule memory works on a prediction of the whole run:

  predict     Qwen drafts its reply with no rules: what it is about to say
  look back   each spot of the draft is a situation = the next AHEAD draft
              tokens (the prediction, END included) + the last RECENT reply
              tokens; the nearest stored moment above tau says "you were
              about to say X here, the rule said Y instead": X is pushed
              down and Y up by their measured log-ratios, the reply is cut
              there and drafted again from the new token

  rehearse (when the rules are given; the same loop, the rules as teacher)
    Qwen drafts each practice answer with no rules; the draft is scored with
    the rules in view; the first spot where the rules would choose another
    token, and either push (Y up, X down) is over GAP nats, is stored as
    (situation, X, Y, down, up); the reply is cut there, Y written, and
    drafted on with no rules. So what is stored are the states the test
    will be in: Qwen's own unruled drafts, earlier corrections applied.
    Moments with the same (X, Y) are bundled into one glyphh (step7b).

  practice    step7's questions, plus for each rule "questions an assistant
              WITHOUT this instruction would answer in a way that breaks it"
              (step7's were about the rule, never where it acts). Any equal
              to a test question is dropped and counted.

Fixed before the run: gap 8 nats, RECENT 3, bundles of 3+ (step6/7), AHEAD 3
weighted 1, 1/2, 1/4, tau from Frankenstein at 99.5%, a bundle changes a
reply at most once.

step8b (after 8a, so a second try on the test questions; both changes come
from the rehearsal log, not the test replies):
  --tempt   the "breaks it" ask got questions about the rule again; ask for
            "user requests that would tempt an assistant to do what the
            instruction forbids" instead (statements are kept, not only
            questions)
  --by-y    bundle by what the rule chose (as step7b did), not by the
            (X, Y) pair: "after Cheers! -> end" was split over many X. The
            token pushed down is then whatever the draft was about to say.

Run:  python step8_lookback.py [--tempt --by-y --out step8b_results.json]
"""

import argparse
import json
import os
import re
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

import qwen
import step3_glyphh as s3
import step6_rules as s6
from step1_adapter import HERE, book
from step5_instructions import BRIEF, GENERAL, ML, RULES, chat, checks
from step7_rehearse import ASKS, RULE_LINES, practice
import step7_rehearse as s7

AHEAD = 3
RECENT = s6.RECENT
MAX_NEW = 90
BREAK_ASKS = [
    f"Here is an instruction for an AI assistant: \"{r}\" Write 30 different short questions "
    f"from users that an assistant WITHOUT this instruction would likely answer in a way that "
    f"breaks it. One per line, no numbering." for r in RULE_LINES]
TEMPT_ASKS = [
    f"Here is an instruction for an AI assistant: \"{r}\" Write 30 different short user requests "
    f"that would tempt an assistant to do what the instruction forbids. One per line, no numbering."
    for r in RULE_LINES]


def tempting(q, have):
    """practice requests from TEMPT_ASKS; statements count, test questions don't."""
    test = {t.lower().strip() for t in GENERAL + ML}
    qs, dropped = [], 0
    for ask in TEMPT_ASKS:
        text, _ = s7.plain_generate(q, chat(q, [{"role": "user", "content": ask}]), max_new=900)
        for line in text.splitlines():
            line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip()
            if len(line) < 10 or line[-1] not in ".?!":
                continue
            if line.lower() in test:
                dropped += 1
            elif line not in qs and line not in have:
                qs.append(line)
    return qs, dropped


def draft(q, prompt_toks, prefix, max_new):
    """greedy, no rules: the tokens Qwen is about to say after prefix (the
    stop token kept as the last one if it gets there) and the logits of each."""
    stop = {q.tok.eos_token_id, q.tok.convert_tokens_to_ids("<|im_end|>")}
    cache = [KVCache() for _ in range(q.n_layers)]
    logits, _ = q.forward(mx.array(np.array(prompt_toks + prefix, np.int32)[None]), (), cache)
    out, lgs = [], []
    for _ in range(max(max_new, 0)):
        lg = np.array(logits[0, -1].astype(mx.float32))
        nxt = int(lg.argmax())
        out.append(nxt)
        lgs.append(lg)
        if nxt in stop:
            break
        logits, _ = q.forward(mx.array([[nxt]], dtype=mx.int32), (), cache)
    return out, lgs


class Lookback(s6.RuleMemory):
    def __init__(self, q, tau=0.5):
        super().__init__(q, tau)
        self.xs, self.ys, self.downs, self.ups = [], [], [], []

    def situation(self, ahead, recent):
        a = sum(0.5 ** k * self.piece(t) for k, t in enumerate(ahead[:AHEAD]))
        a = a / (np.linalg.norm(a) or 1.0)
        start = s3.symbol("rule:START", s3.SEED, s3.DIM).astype(np.float32)
        r = sum(self.piece(t) for t in recent[-RECENT:]) if recent else start
        r = r / (np.linalg.norm(r) or 1.0)
        v = a + r
        return v / (np.linalg.norm(v) or 1.0)

    def scored(self, toks, start, n):
        lg, _ = self.q.forward(mx.array(np.array(toks, np.int32)[None]))
        lg = lg[0, start:start + n].astype(mx.float32)
        return np.array(lg - mx.logsumexp(lg, axis=-1, keepdims=True), np.float64)

    def rehearse(self, rules_prompt, plain_prompt, gap, max_edits=8):
        """one practice answer: draft with no rules, the rules correct it."""
        rp, pp = self.q.encode(rules_prompt), self.q.encode(plain_prompt)
        prefix, d = [], []
        for _ in range(max_edits):
            d, lgs = draft(self.q, pp, prefix, MAX_NEW - len(prefix))
            if not d:
                break
            lr = self.scored(rp + prefix + d, len(rp) + len(prefix) - 1, len(d))
            cut = None
            for i, x in enumerate(d):
                y = int(lr[i].argmax())
                if y == x:
                    continue
                lp = lgs[i].astype(np.float64)
                lp -= np.logaddexp.reduce(lp)
                up, down = lr[i][y] - lp[y], lp[x] - lr[i][x]
                if max(up, down) > gap:
                    cut = (i, y)
                    self.keys.append(self.situation(d[i:], prefix + d[:i]))
                    self.xs.append(x)
                    self.ys.append(y)
                    self.downs.append(float(down))
                    self.ups.append(float(up))
                    self.notes.append((self.q.tok.decode((prefix + d[:i])[-6:]), self.q.tok.decode([x]),
                                       self.q.tok.decode([y]), round(float(down), 1), round(float(up), 1)))
                    break
            if cut is None:
                break
            i, y = cut
            prefix, d = prefix + d[:i], []
            if y in self.end:
                break
            prefix.append(y)
        return self.q.tok.decode([t for t in prefix + d if t not in self.end]).strip()

    def bundle(self, min_n=3, by_y=False):
        by = {}
        for j, xy in enumerate(zip(self.xs, self.ys)):
            by.setdefault((None, xy[1]) if by_y else xy, []).append(j)
        keys, xs, ys, downs, ups, notes = [], [], [], [], [], []
        for (x, y), js in by.items():
            if len(js) < min_n:
                continue
            v = np.sum([self.keys[j] for j in js], axis=0)
            keys.append(v / np.linalg.norm(v))
            xs.append(x)
            ys.append(y)
            downs.append(float(np.median([self.downs[j] for j in js])))
            ups.append(float(np.median([self.ups[j] for j in js])))
            notes.append(("*" if x is None else self.q.tok.decode([x]), self.q.tok.decode([y]), len(js),
                          round(downs[-1], 1), round(ups[-1], 1)))
        self.keys, self.xs, self.ys, self.downs, self.ups = keys, xs, ys, downs, ups
        return notes

    def calibrate(self, pct=99.5, per_gate=False):
        """per_gate (step8e): each gate gets its own tau, what unrelated text
        reaches on THAT gate; one tau for all is set by the gates on common
        words and leaves the ' logistic' gates just under it (diag_rehalt.py)."""
        toks = self.q.encode(book("frankenstein", 84)[50000:54000])
        K = np.array(self.keys)
        s = np.array([K @ self.situation(toks[i:], toks[:i]) for i in range(RECENT, len(toks), 2)])
        return np.percentile(s, pct, axis=0) if per_gate else float(np.percentile(s.max(1), pct))

    def reply(self, prompt, rehalt=False):
        """predict, look back, correct, predict again. rehalt (step8d): a gate
        halts the reply every time the prediction matches it, not once; it
        only sits out while its own token is among the last RECENT written."""
        pt = self.q.encode(prompt)
        K = np.array(self.keys)
        prefix, spent, d, halts = [], set(), [], 0
        while True:
            d, lgs = draft(self.q, pt, prefix, MAX_NEW - len(prefix))
            cut = None
            for i in range(len(d)):
                s = K @ self.situation(d[i:], prefix + d[:i])
                if rehalt:
                    said = set((prefix + d[:i])[-RECENT:])
                    spent = {j for j, y in enumerate(self.ys) if y in said}
                s = s - self.tau
                s[list(spent)] = -1
                j = int(s.argmax())
                if s[j] < 0:
                    continue
                lg = lgs[i].astype(np.float64)
                lg[d[i] if self.xs[j] is None else self.xs[j]] -= self.downs[j]
                lg[self.ys[j]] += self.ups[j]
                nxt = int(lg.argmax())
                if nxt != d[i]:
                    spent.add(j)
                    halts += 1
                    cut = (i, nxt)
                    break
            if cut is None:
                break
            i, nxt = cut
            prefix, d = prefix + d[:i], []
            if nxt in self.end:
                break
            prefix.append(nxt)
        return self.q.tok.decode([t for t in prefix + d if t not in self.end]).strip(), halts


def main(gap, show, out, tempt=False, by_y=False, rehalt=False, per_gate=False):
    q = qwen.Qwen()
    t = time.time()
    s7.ASKS = ASKS if tempt else ASKS + BREAK_ASKS
    qs, dropped = practice(q)
    if tempt:
        more, d2 = tempting(q, qs)
        qs, dropped = qs + more, dropped + d2
    print(f"{qwen.NAME}; {len(qs)} practice questions written by Qwen ({dropped} identical to a test "
          f"question, dropped) in {time.time() - t:.0f}s, e.g.:")
    for x in qs[::max(1, len(qs) // 10)][:10]:
        print("   ", x)
    rules = Lookback(q)
    t = time.time()
    taught = []
    for question in qs:
        rp = chat(q, [{"role": "system", "content": BRIEF + " " + RULES}, {"role": "user", "content": question}])
        pp = chat(q, [{"role": "system", "content": BRIEF}, {"role": "user", "content": question}])
        taught.append((question, rules.rehearse(rp, pp, gap)))
    points = list(rules.notes)
    print(f"rehearsed in {time.time() - t:.0f}s (corrected drafts end with 'Cheers!' "
          f"{np.mean([r.rstrip().endswith('Cheers!') for _, r in taught]):.0%}, say logistic "
          f"{np.mean(['logistic' in r.lower() for _, r in taught]):.0%}): {len(points)} moments")
    b = rules.bundle(by_y=by_y)
    print("  bundled: " + ", ".join(f"{x!r}->{y!r} x{n} (-{d} +{u})" for x, y, n, d, u in
                                    sorted(b, key=lambda r: -r[2])))
    rules.tau = rules.calibrate(per_gate=per_gate)
    print("  tau " + (" ".join(f"{t:.2f}" for t in rules.tau) if per_gate else f"{rules.tau:.3f}") + "\n")

    results = {"practice": qs, "rehearsed": taught, "points": points, "bundles": b, "tau": np.asarray(rules.tau).tolist()}
    for arm in ("alone", "prompt", "lookback"):
        system = BRIEF + (" " + RULES if arm == "prompt" else "")
        rows, t = [], time.time()
        for question in GENERAL + ML:
            prompt = chat(q, [{"role": "system", "content": system}, {"role": "user", "content": question}])
            reply, fired = rules.reply(prompt, rehalt) if arm == "lookback" else s6.generate(q, prompt)
            rows.append({"q": question, "reply": reply, "fired": fired, **checks(reply, question in ML)})
        rate = lambda k, rs=rows: np.mean([r[k] for r in rs if k in r])  # noqa: E731
        print(f"{arm:<8} signoff {rate('signoff'):4.0%}  ban {rate('ban'):4.0%}  (says logistic "
              f"{rate('logistic'):4.0%})  corrections per reply {rate('fired'):.1f}  ({time.time() - t:.0f}s)")
        for r in rows[:show] + [r for r in rows if r["q"] in ML][:show]:
            print(f"    {r['q']} -> {r['reply']!r}")
        results[arm] = rows
        with open(os.path.join(HERE, out), "w") as fh:
            json.dump(results, fh, indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=8.0)
    ap.add_argument("--show", type=int, default=4)
    ap.add_argument("--out", default="step8_results.json")
    ap.add_argument("--tempt", action="store_true", help="step8b: practice requests that tempt the rule")
    ap.add_argument("--by-y", action="store_true", help="step8b: bundle by what the rule chose")
    ap.add_argument("--rehalt", action="store_true", help="step8d: a gate halts every time it matches")
    ap.add_argument("--per-gate", action="store_true", help="step8e: a tau per gate")
    a = ap.parse_args()
    main(a.gap, a.show, a.out, a.tempt, a.by_y, a.rehalt, a.per_gate)
