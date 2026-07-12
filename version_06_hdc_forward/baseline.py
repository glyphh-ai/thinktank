"""
baseline.py — the SACRED baseline: vanilla word-level GPT on the big corpus.
===========================================================================
Per the experimental plan in adaL/docs/glyph-ada-training-loop.md:

  "Baseline (sacred): vanilla GPT on the corpus. Same tokens, same steps, same
   seed. This delta is the whole claim."

No HDC, no role head, no lattice — just next-word prediction. We report
perplexity (the LM number the treatment must not wreck) and show free-running
generations so we can see, qualitatively, where an ungrounded model drifts:
it will happily emit fluent-but-false facts (wrong city, wrong element). That
drift is exactly what the HDC forward-pass gate (forward_hdc.py) is meant to catch.

Run:  python baseline.py
"""
from __future__ import annotations

import numpy as np

import model as M
from corpus import build_corpus, SEP

T = 16          # context window (words)
C = 64          # embed dim
STEPS = 2000
SEED = 0


def generate(p, stoi, itos, prompt_words, n=10, T=T, temp=0.0, seed=1):
    rng = np.random.default_rng(seed)
    ctx = [stoi.get(w, stoi.get("<unk>", 0)) for w in prompt_words]
    out = list(prompt_words)
    for _ in range(n):
        logits = M.next_logits(p, ctx, T)
        if temp <= 0:
            nxt = int(logits.argmax())
        else:
            z = logits / temp
            z -= z.max()
            pr = np.exp(z); pr /= pr.sum()
            nxt = int(rng.choice(len(pr), p=pr))
        w = itos[nxt]
        out.append(w)
        ctx.append(nxt)
        if w == SEP:
            break
    return out


if __name__ == "__main__":
    print("building corpus (both sides of the lattice)...")
    c = build_corpus(repeat=60)
    ids, vocab, stoi = c["ids"], c["vocab"], c["stoi"]
    itos = {i: w for w, i in stoi.items()}
    n = int(0.9 * len(ids))
    tr, va = ids[:n], ids[n:]
    print(f"  tokens={len(ids):,}  vocab={len(vocab)}  train={len(tr):,}  val={len(va):,}\n")

    print(f"training SACRED baseline (vanilla LM, {STEPS} steps, seed {SEED})...")
    p = M.train(tr, len(vocab), T=T, C=C, steps=STEPS, seed=SEED)

    rng = np.random.default_rng(123)
    tr_ppl = np.exp(M.eval_ppl(p, tr, T, rng))
    va_ppl = np.exp(M.eval_ppl(p, va, T, rng))
    print(f"\nperplexity  train={tr_ppl:.2f}  val={va_ppl:.2f}  (the number the HDC treatment must not wreck)\n")

    print("free-running greedy generations (prompt -> continuation):")
    for prompt in (["carol", "lives", "in"], ["helium", "is", "a"],
                   ["boston", "is", "in"], ["jack", "works", "as", "a"],
                   ["gold", "looks"]):
        cont = generate(p, stoi, itos, prompt, n=6)
        print(f"  {' '.join(prompt):24} -> {' '.join(cont[len(prompt):])}")

    # save weights for forward_hdc.py to reuse (no retrain needed)
    np.savez("baseline_ckpt.npz", **{k: v for k, v in p.items()},
             _vocab=np.array(vocab, dtype=object))
    print("\nsaved baseline_ckpt.npz")
