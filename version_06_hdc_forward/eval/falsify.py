"""
falsify.py — the make-or-break test for "grounded generation beats a DB".
=========================================================================
For span-level grounded generation to give a HARD guarantee (0 ungated
hallucination), we must DETECT every factual assertion in free text so we can gate
it. If detection recall < ~100%, ungated assertions leak and the guarantee dies —
so "better than a DB" is refuted.

This measures an UPPER BOUND on that recall: the easiest possible case —
  - ONE fact per sentence (real PopQA triple),
  - a clean model-written sentence conveying it,
  - the strongest extractor we have (Qwen3-4B) doing open-IE on it.
If recall here isn't ~100%, real multi-assertion generation is only worse.

recall = fraction of sentences where the extractor recovers the known assertion.
Also reports assertion density (triples/sentence) -> the per-sentence attack surface,
and the implied P(all caught) for a k-assertion paragraph.

Run:  /opt/homebrew/anaconda3/bin/python falsify.py
"""
from __future__ import annotations

import os
import re

from mlx_lm import load, generate

import popqa

SP = os.environ.get("SCRATCH", "/private/tmp/claude-501/-Users-timmetim-development-yo/"
                    "88a38df1-4acb-4e03-aa78-d95dcbc91eb9/scratchpad")
MODEL = "mlx-community/Qwen3-4B-4bit"     # strongest local extractor -> best-case recall
N = 80


def _chat(tok, msg):
    try:
        return tok.apply_chat_template([{"role": "user", "content": msg}],
                                       add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template([{"role": "user", "content": msg}],
                                       add_generation_prompt=True, tokenize=False)


def gen(model, tok, msg, mx=80):
    out = generate(model, tok, prompt=_chat(tok, msg), max_tokens=mx, verbose=False)
    return out.replace("<think>", " ").replace("</think>", " ").strip()


def main():
    items = popqa.load(f"{SP}/popQA.tsv")
    picked = popqa.sample(items, per_tercile=N // 3)[:N]
    model, tok = load(MODEL)

    detected = 0
    density = []
    misses = []
    for i, it in enumerate(picked):
        sent = gen(model, tok, f"Write one natural-sounding sentence conveying exactly this fact: "
                               f"the {it.prop} of {it.subj} is {it.obj}. Output only the sentence.",
                   mx=48).split("\n")[0].strip()
        extraction = gen(model, tok, "Extract EVERY factual claim in the sentence as triples, one per "
                                    "line, formatted 'subject | relation | object'. Sentence:\n" + sent, mx=120)
        # assertion density = non-empty triple lines
        lines = [l for l in extraction.splitlines() if "|" in l]
        density.append(len(lines))
        # SURFACE recall: the fact is detected if the gold object appears ANYWHERE in
        # an extracted triple (direction/relation-naming agnostic — the generous case).
        hit = any(it.obj.lower() in l.lower() for l in lines)
        detected += hit
        if not hit:
            misses.append((it.subj, it.prop, it.obj, sent[:70], extraction[:90].replace("\n", " | ")))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(picked)}")

    n = len(picked)
    recall = detected / n
    avg_density = sum(density) / n
    print("\n" + "=" * 74)
    print(f"ASSERTION-DETECTION RECALL (upper bound; {MODEL}, N={n})")
    print("=" * 74)
    print(f"recall (known 1-fact assertion detected): {recall:.1%}  ({detected}/{n})")
    print(f"assertion density (triples/sentence)    : {avg_density:.1f}")
    print("\nimplied P(ALL assertions caught) in a k-assertion paragraph at this recall:")
    for k in (1, 3, 5, 8):
        print(f"    k={k}: {recall**k:5.1%}")
    print("\n(a HARD 0-hallucination guarantee needs ~100%; anything less leaks in multi-fact text.)")
    print("\nsample misses (assertion present but NOT detected):")
    for s, p, o, sent, ext in misses[:6]:
        print(f"  [{p}={o}] sent={sent!r}")
        print(f"           extracted: {ext}")


if __name__ == "__main__":
    main()
