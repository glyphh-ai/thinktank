"""step5_instructions.py — do instructions carry across sessions in the glyphh?

Session 1: the user gives three rules once, then asks six questions; Qwen
answers them with the rules in front of it (that chat is the teaching). The
whole session-1 transcript is learned into the glyphh memory (step4c's: delta
0.05 mixing, function words out of the key).

Session 2: a fresh chat, new questions it has never seen, NO rules in the
prompt. Three arms, greedy answers:
  qwen alone       no rules, no memory
  rules in prompt  the rules pasted into the system prompt (the usual way)
  glyphh memory    no rules in the prompt; the memory taught on session 1

Rules and checks (every arm's system prompt also says "Answer in two
sentences or fewer.", so answers stay short):
  name     "Call me Chris."                         reply contains "Chris"
  signoff  "End every answer with 'Cheers!'"        reply ends with "Cheers!"
  ban      "Never recommend logistic regression;    ML questions only: reply says
            recommend a decision tree instead."     "decision tree", never "logistic"

Run:  python step5_instructions.py [--arms alone,prompt,glyphh] [--show 3]
"""

import argparse
import json
import os
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

import qwen
import step3_glyphh as s3
from step1_adapter import HERE, book

s3.STOP |= s3.FUNCTION
BRIEF = "Answer in two sentences or fewer."
RULES = ("Call me Chris. End every answer with 'Cheers!'. Never recommend "
         "logistic regression; recommend a decision tree instead.")
OPEN = "A few rules for this whole chat: " + RULES

SESSION1 = [
    "What is a good way to start learning to cook?",
    "How should I predict which customers will cancel their subscription?",
    "Why is the sky blue?",
    "What model would you use to flag fraudulent transactions?",
    "How do I keep basil alive indoors?",
    "Which classifier should I try first for spam detection?",
]
GENERAL = [
    "How long should I boil an egg?", "What causes the seasons?",
    "How do I fix a squeaky door?", "What is a good first houseplant?",
    "How can I sleep better?", "Why do cats purr?",
    "How do I clean a cast iron pan?", "What is the capital of Australia?",
    "How do I get better at chess?", "Why does bread go stale?",
    "How often should I water a cactus?", "What makes a good password?",
    "How do I remove a coffee stain?", "Why is the ocean salty?",
    "How do I start running?", "What is compound interest?",
    "How can I learn a language faster?", "Why do leaves change color?",
    "How do I make cold brew coffee?", "What is a black hole?",
]
ML = [
    "What model should I use to predict whether a loan will default?",
    "How should I classify emails as urgent or not?",
    "Which model predicts if a patient will be readmitted?",
    "What should I use to predict whether a user clicks an ad?",
    "How do I predict if an employee will quit?",
    "What classifier should I use for a yes/no outcome from tabular data?",
    "Which model is best to predict whether a machine will fail this week?",
    "How should I model whether a lead converts to a sale?",
    "What method predicts if a transaction is a chargeback?",
    "Which algorithm should I start with to predict churn?",
    "How do I predict whether a student passes an exam?",
    "What model should I use to tell if a review is positive or negative?",
    "How should I predict if a flight will be delayed?",
    "What is a good baseline model for binary classification?",
    "Which model predicts whether a customer buys again?",
    "How do I predict if a website visitor signs up?",
    "What model should I use to detect whether a tumor is malignant?",
    "How do I estimate whether a borrower repays on time?",
    "Which classifier would you pick to predict voter turnout?",
    "What should I use to predict whether a package arrives late?",
]


def chat(q, messages, gen=True):
    return q.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen,
                                     enable_thinking=False)


def generate(q, prompt, mem=None, max_new=90):
    """greedy; with mem, the glyphh memory mixes into every next-token distribution."""
    toks = q.encode(prompt)
    stop = {q.tok.eos_token_id, q.tok.convert_tokens_to_ids("<|im_end|>")}
    cache = [KVCache() for _ in range(q.n_layers)]
    logits, _ = q.forward(mx.array(np.array(toks, np.int32)[None]), (), cache)
    text, out, fired = prompt, [], 0
    for _ in range(max_new):
        lg = np.array(logits[0, -1].astype(mx.float32), np.float64)
        p = np.exp(lg - lg.max())
        p /= p.sum()
        if mem is not None:
            p2, _ = mem.next_probs(text, p)
            fired += p2 is not p
            p = p2
        nxt = int(p.argmax())
        if nxt in stop:
            break
        out.append(nxt)
        text += q.tok.decode([nxt])
        logits, _ = q.forward(mx.array([[nxt]], dtype=mx.int32), (), cache)
    return q.tok.decode(out).strip(), fired / max(len(out), 1)


def checks(reply, ml):
    r = {"name": "chris" in reply.lower(),
         "signoff": reply.rstrip().endswith("Cheers!")}
    if ml:
        low = reply.lower()
        r["ban"] = "decision tree" in low and "logistic" not in low
        r["logistic"] = "logistic" in low
    return r


def session1(q):
    """the teaching chat: rules once, six questions answered with the rules in view."""
    msgs = [{"role": "system", "content": BRIEF}, {"role": "user", "content": OPEN + "\n\n" + SESSION1[0]}]
    replies = []
    for i, question in enumerate(SESSION1):
        if i:
            msgs.append({"role": "user", "content": question})
        reply, _ = generate(q, chat(q, msgs))
        replies.append((question, reply, checks(reply, i % 2 == 1)))
        msgs.append({"role": "assistant", "content": reply})
    return chat(q, msgs, gen=False), replies


def calibrate(q, mem, pct=99.5):
    other = q.encode(book("frankenstein", 84)[50000:58000])
    _, s = mem.nearest(mem.contexts(other))
    return float(np.percentile(s, pct))


def main(arms, show, out):
    q = qwen.Qwen()
    t = time.time()
    transcript, taught = session1(q)
    print(f"{qwen.NAME}; session 1 ({time.time() - t:.0f}s), rules followed in view:")
    for question, reply, c in taught:
        print(f"  {' '.join(f'{k} {v:d}' for k, v in c.items() if k != 'logistic')}  | {question} -> {reply!r}")
    mem = s3.GlyphhMemory(q.tok, delta=0.05)
    mem.learn(q.encode(transcript))
    mem.tau = calibrate(q, mem)
    print(f"  memory: {len(mem.counts)} keys from the session-1 transcript, tau {mem.tau:.3f}\n")

    results = {"session1": [(a, b, c) for a, b, c in taught], "tau": mem.tau}
    pp = book("pride_and_prejudice", 1342)[:20000]
    for arm in arms:
        system = BRIEF + (" " + RULES if arm == "prompt" else "")
        m = mem if arm == "glyphh" else None
        rows, t = [], time.time()
        for question in GENERAL + ML:
            prompt = chat(q, [{"role": "system", "content": system}, {"role": "user", "content": question}])
            reply, fired = generate(q, prompt, m)
            rows.append({"q": question, "reply": reply, "fired": fired, **checks(reply, question in ML)})
        rate = lambda k, rs=rows: np.mean([r[k] for r in rs if k in r])  # noqa: E731
        print(f"{arm:<7} name {rate('name'):4.0%}  signoff {rate('signoff'):4.0%}  ban {rate('ban'):4.0%}"
              f"  (says logistic {rate('logistic'):4.0%})  memory fired on {rate('fired'):4.0%} of tokens"
              f"  ({time.time() - t:.0f}s)")
        for r in rows[:show] + [r for r in rows if r["q"] in ML][:show]:
            print(f"    {r['q']} -> {r['reply']!r}")
        results[arm] = rows
        if arm == "glyphh":
            toks = q.encode(pp)
            lp = q.run([toks], batch=1)[0][0]
            base = -lp.sum() / np.log(2) / len(pp)
            lp2 = mem.refine(toks, lp)
            results["pp_bpc"] = (base, -lp2.sum() / np.log(2) / len(pp))
            print(f"    English (P&P bits/char): qwen {base:.3f}, with memory {results['pp_bpc'][1]:.3f}")
        with open(os.path.join(HERE, out), "w") as fh:
            json.dump(results, fh, indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="alone,prompt,glyphh")
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--out", default="step5_results.json")
    a = ap.parse_args()
    main(a.arms.split(","), a.show, a.out)
