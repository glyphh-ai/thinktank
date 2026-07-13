"""
run_eval.py — grounding evaluation on PopQA: closed-book vs RAG vs gate, on Qwen3.
=================================================================================
Claim under test: the grounding GATE eliminates hallucination on out-of-KB queries
(provable abstention) while matching RAG in-KB, at negligible cost, size-independent.

Metrics:
  IN-KB   (fact stored)      : accuracy (answered & correct), by popularity tercile
  OUT-KB  (real, not stored) : abstention, and HALLUCINATION = asserted-and-wrong
  reliability = 1 - (confidently-wrong rate)   [never asserts something false]

Run:  /opt/homebrew/anaconda3/bin/python run_eval.py --model mlx-community/Qwen3-1.7B-4bit --per-tercile 40
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

from mlx_lm import load

import popqa
import methods as M

SP = os.environ.get("SCRATCH", "/private/tmp/claude-501/-Users-timmetim-development-yo/"
                    "88a38df1-4acb-4e03-aa78-d95dcbc91eb9/scratchpad")


def outcome(pred, item):
    abst = M.is_abstain(pred)
    corr = (not abst) and popqa.matches(pred, item.answers)
    return abst, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    ap.add_argument("--per-tercile", type=int, default=40)
    ap.add_argument("--out", default=f"{SP}/eval_results.jsonl")
    args = ap.parse_args()

    items = popqa.load(f"{SP}/popQA.tsv")
    picked = popqa.sample(items, per_tercile=args.per_tercile)
    kb, held = popqa.split_by_entity(picked)
    store = M.build_store(kb)
    retr = M.Retriever([popqa.render_passage(it) for it in kb])
    print(f"model={args.model}  KB={len(kb)}  heldout={len(held)}  (per_tercile={args.per_tercile})")

    model, tok = load(args.model)

    # accumulators: stats[method][population] -> counters
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    terc = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # in-KB accuracy by tercile
    lat = defaultdict(float); nlat = defaultdict(int)

    def run(pop_name, batch, in_kb):
        for i, it in enumerate(batch):
            for name, fn in (("closed", lambda: M.closed_book(model, tok, it.question)),
                             ("rag", lambda: M.rag(model, tok, it.question, retr)),
                             ("gate", lambda: M.gate(model, tok, it.question, it.subj, it.prop, store))):
                t0 = time.time()
                pred = fn()
                lat[name] += time.time() - t0; nlat[name] += 1
                abst, corr = outcome(pred, it)
                s = stats[name][pop_name]
                s["n"] += 1; s["abstain"] += abst; s["correct"] += corr
                s["asserted"] += (not abst); s["assert_wrong"] += ((not abst) and not corr)
                if in_kb:
                    terc[name][it.tercile]["n"] += 1
                    terc[name][it.tercile]["correct"] += corr
            if (i + 1) % 20 == 0:
                print(f"  {pop_name}: {i+1}/{len(batch)}")

    run("in_kb", kb, True)
    run("out_kb", held, False)

    # ---- report ----
    def pct(a, b): return f"{100*a/b:5.1f}%" if b else "   -  "
    print("\n" + "=" * 78)
    print(f"RESULTS  {args.model}")
    print("=" * 78)
    print(f"{'':8} | {'IN-KB acc':>10} {'in-KB abst':>11} | {'OUT abstain':>12} "
          f"{'OUT halluc':>11} {'reliability':>12}")
    for name in ("closed", "rag", "gate"):
        ik, ok = stats[name]["in_kb"], stats[name]["out_kb"]
        rel_wrong = ik["assert_wrong"] + ok["assert_wrong"]
        rel = 1 - rel_wrong / (ik["n"] + ok["n"])
        print(f"{name:8} | {pct(ik['correct'], ik['n']):>10} {pct(ik['abstain'], ik['n']):>11} | "
              f"{pct(ok['abstain'], ok['n']):>12} {pct(ok['assert_wrong'], ok['n']):>11} {100*rel:11.1f}%")
    print("\nIN-KB accuracy by popularity tercile (rare/mid/popular):")
    for name in ("closed", "rag", "gate"):
        cells = " ".join(f"{t}={pct(terc[name][t]['correct'], terc[name][t]['n'])}"
                         for t in ("rare", "mid", "popular"))
        print(f"  {name:8} {cells}")
    print("\nlatency/query:", " ".join(f"{n}={1000*lat[n]/max(1,nlat[n]):.0f}ms" for n in ("closed", "rag", "gate")))

    row = {"model": args.model, "kb": len(kb), "heldout": len(held),
           "stats": {n: {p: dict(stats[n][p]) for p in stats[n]} for n in stats},
           "tercile": {n: {t: dict(terc[n][t]) for t in terc[n]} for n in terc},
           "latency_ms": {n: 1000 * lat[n] / max(1, nlat[n]) for n in lat}}
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
