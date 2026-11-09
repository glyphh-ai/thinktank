"""step6_rules.py — instructions as their own channel: rule nudges in the glyphh.

step5 showed the fact memory can't carry a rule: keyed on the last 8 words,
it replays session-1 sentences. A rule is different: "when you're about to
do X, do Y instead". So:

  learn (session 1, once)
    for every session-1 reply (written with the rules in view), run Qwen
    again WITHOUT the rules on the same reply; where the rules made the
    chosen token far likelier (log-ratio > GAP nats) is a decision point:
    the rule acted there. Only those are stored, nothing else of the chat.
    key   = the situation: a glyphh of what Qwen-without-rules was about to
            do there (its top-5 next tokens, as word glyphs weighted by
            probability; ending the reply is the symbol END)
    value = the token the rule chose instead

  apply (session 2, no rules in the prompt)
    at each step, the glyphh of what Qwen is about to do; a stored decision
    point above tau mixes its token in (the step3 counter mixing, beta 1).
    Each decision point fires at most once per reply (otherwise "about to
    end -> add 'Cheers!'" fires again after the 'Cheers!').

So the rule fires on what the model is about to do, not on the words around
it: about to say "logistic" -> "decision"; about to stop -> "Cheers!".

Run:  python step6_rules.py [--gap 2.0] [--show 4]
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
from step1_adapter import HERE, book
from step5_instructions import BRIEF, GENERAL, ML, RULES, chat, checks, session1

TOPK = 5
RECENT = 3
_W = re.compile(r"[a-z]+")


class RuleMemory:
    def __init__(self, q, tau=0.5, beta=1.0):
        self.q, self.tau, self.beta = q, tau, beta
        self.g = s3.GlyphhMemory(q.tok)            # for its word glyphs
        self.end = {q.tok.eos_token_id, q.tok.convert_tokens_to_ids("<|im_end|>")}
        self.keys, self.tokens, self.gaps, self.notes = [], [], [], []

    def piece(self, t):
        """glyph of one token: its words, or END / a punctuation mark / BREAK."""
        if t in self.end:
            return s3.symbol("rule:END", s3.SEED, s3.DIM).astype(np.float32)
        piece = self.q.tok.decode([int(t)])
        words = _W.findall(piece.lower())
        if words:
            return sum(self.g._wglyph(w) for w in words)
        if piece.strip():
            return s3.symbol(f"rule:punct:{piece.strip()}", s3.SEED, s3.DIM).astype(np.float32)
        return s3.symbol("rule:BREAK", s3.SEED, s3.DIM).astype(np.float32)

    def situation(self, logits, recent):
        """glyphh of the situation: what Qwen is about to do (its top-k next
        tokens, weighted) plus the last RECENT tokens of the reply so far (so
        "about to say ' to'" anywhere isn't the same as after "Use a")."""
        z = logits - logits.max()
        p = np.exp(z)
        p /= p.sum()
        top = np.argpartition(-p, TOPK)[:TOPK]
        v = sum(float(p[t]) * self.piece(t) for t in top)
        v = v / (np.linalg.norm(v) or 1.0)
        start = s3.symbol("rule:START", s3.SEED, s3.DIM).astype(np.float32)
        r = sum(self.piece(t) for t in recent[-RECENT:]) if recent else start
        r = r / (np.linalg.norm(r) or 1.0)
        v = v + r
        return v / (np.linalg.norm(v) or 1.0)

    def logits(self, toks):
        lg, _ = self.q.forward(mx.array(np.array(toks, np.int32)[None]))
        return np.array(lg[0].astype(mx.float32), np.float64)

    def learn(self, rules_prompt, plain_prompt, reply, gap):
        """reply was written under rules_prompt; store where the rules decided it."""
        rp, pp = self.q.encode(rules_prompt), self.q.encode(plain_prompt)
        r = self.q.encode(reply) + [self.q.tok.convert_tokens_to_ids("<|im_end|>")]
        lr, lp = self.logits(rp + r), self.logits(pp + r)
        for i, tok in enumerate(r):
            a, b = lr[len(rp) - 1 + i], lp[len(pp) - 1 + i]
            ra = a[tok] - np.logaddexp.reduce(a)
            rb = b[tok] - np.logaddexp.reduce(b)
            if ra - rb > gap:
                self.keys.append(self.situation(b, r[:i]))
                self.tokens.append(tok)
                self.gaps.append(float(ra - rb))
                was = self.q.tok.decode([int(b.argmax())])
                self.notes.append((self.q.tok.decode(r[max(0, i - 6):i]), was,
                                   self.q.tok.decode([tok]), round(float(ra - rb), 1)))

    def nudge(self, logits, recent, p, spent):
        if not self.keys:
            return p, None
        K = np.array(self.keys)
        s = K @ self.situation(logits, recent)
        s[list(spent)] = -1
        j = int(s.argmax())
        if s[j] < self.tau:
            return p, None
        # the rule's own measured effect: shift its token by the log-ratio it
        # had in session 1 (count mixing at beta 1 can't beat a 95% token)
        lg = np.log(np.maximum(p, 1e-30))
        lg[self.tokens[j]] += self.gaps[j]
        q = np.exp(lg - lg.max())
        return q / q.sum(), j


def generate(q, prompt, rules=None, max_new=90):
    toks = q.encode(prompt)
    stop = {q.tok.eos_token_id, q.tok.convert_tokens_to_ids("<|im_end|>")}
    cache = [KVCache() for _ in range(q.n_layers)]
    logits, _ = q.forward(mx.array(np.array(toks, np.int32)[None]), (), cache)
    out, spent = [], set()
    for _ in range(max_new):
        lg = np.array(logits[0, -1].astype(mx.float32), np.float64)
        p = np.exp(lg - lg.max())
        p /= p.sum()
        if rules is not None:
            p, j = rules.nudge(lg, out, p, spent)
            if j is not None:
                spent.add(j)
        nxt = int(p.argmax())
        if nxt in stop:
            break
        out.append(nxt)
        logits, _ = q.forward(mx.array([[nxt]], dtype=mx.int32), (), cache)
    return q.tok.decode(out).strip(), len(spent)


def calibrate(q, rules, pct=99.5):
    """tau: what unrelated text's situations reach (Frankenstein, rules never apply)."""
    toks = q.encode(book("frankenstein", 84)[50000:54000])
    lg = rules.logits(toks)
    K = np.array(rules.keys)
    s = [float((K @ rules.situation(lg[i], toks[max(0, i - RECENT + 1):i + 1])).max())
         for i in range(0, len(toks), 2)]
    return float(np.percentile(s, pct))


def main(gap, show, out):
    q = qwen.Qwen()
    _, taught = session1(q)
    rules = RuleMemory(q)
    hist = [{"role": "system", "content": BRIEF}]
    for i, (question, reply, _) in enumerate(taught):
        user = ("A few rules for this whole chat: " + RULES + "\n\n" + question) if i == 0 else question
        hist.append({"role": "user", "content": user})
        plain = chat(q, [{"role": "system", "content": BRIEF}, {"role": "user", "content": question}])
        rules.learn(chat(q, hist), plain, reply, gap)
        hist.append({"role": "assistant", "content": reply})
    print(f"{qwen.NAME}; {len(rules.keys)} decision points from session 1 (gap {gap} nats):")
    for before, was, chose, d in rules.notes:
        print(f"  ...{before!r}: without rules {was!r}, rules chose {chose!r} (+{d})")
    rules.tau = calibrate(q, rules)
    print(f"  tau {rules.tau:.3f}\n")

    results = {"points": rules.notes, "tau": rules.tau}
    for arm in ("alone", "prompt", "rules"):
        system = BRIEF + (" " + RULES if arm == "prompt" else "")
        rows, t = [], time.time()
        for question in GENERAL + ML:
            prompt = chat(q, [{"role": "system", "content": system}, {"role": "user", "content": question}])
            reply, fired = generate(q, prompt, rules if arm == "rules" else None)
            rows.append({"q": question, "reply": reply, "fired": fired, **checks(reply, question in ML)})
        rate = lambda k, rs=rows: np.mean([r[k] for r in rs if k in r])  # noqa: E731
        print(f"{arm:<7} signoff {rate('signoff'):4.0%}  ban {rate('ban'):4.0%}  (says logistic "
              f"{rate('logistic'):4.0%})  nudges per reply {rate('fired'):.1f}  ({time.time() - t:.0f}s)")
        for r in rows[:show] + [r for r in rows if r["q"] in ML][:show]:
            print(f"    {r['q']} -> {r['reply']!r}")
        results[arm] = rows
        with open(os.path.join(HERE, out), "w") as fh:
            json.dump(results, fh, indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=2.0)
    ap.add_argument("--show", type=int, default=4)
    ap.add_argument("--out", default="step6_results.json")
    a = ap.parse_args()
    main(a.gap, a.show, a.out)
