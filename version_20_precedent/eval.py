"""eval.py — one arm on both test sets, greedy, scored by decision and by move.

Run: python eval.py --arm move [--adapter adapters/move] [--model MODEL] [--n 600]
Leaving --adapter out scores the base model zero-shot.
"""

import argparse
import collections
import json
import os
import re
import time

from mlx_lm import generate, load

HERE = os.path.dirname(os.path.abspath(__file__))
ZERO = ("You rule on cases from precedent. Precedents marked failed were decided wrongly. "
        "A policy may have changed over time. End with 'decision: ' and one of the two decisions.\n\n")


def parse(text):
    body = text.split("</think>")[-1]
    m = re.search(r"decision:\s*(\w+)", body)
    return (m.group(1) if m else None), body.strip()


def main(a):
    model, tok = load(a.model, adapter_path=a.adapter)
    out = {}
    for name in ("unseen", "harder"):
        rows = json.load(open(f"{HERE}/data/test_{name}.json"))
        rows = rows[:a.n]
        hit, moved, n, t = collections.Counter(), collections.Counter(), collections.Counter(), time.time()
        log = []
        for s in rows:
            content = s["prompt"] if a.adapter else ZERO + s["prompt"]
            p = tok.apply_chat_template([{"role": "user", "content": content}],
                                        add_generation_prompt=True, tokenize=False,
                                        **({} if a.adapter else {"enable_thinking": False}))
            text = generate(model, tok, prompt=p, max_tokens=a.max_tokens, verbose=False)
            d, body = parse(text)
            k = s["kind"]
            n[k] += 1
            hit[k] += d == s["decision"]
            moved[k] += s["move"] in body
            log.append(dict(kind=k, gold=s["decision"], move=s["move"], got=body))
        total = sum(n.values())
        row = dict(all=sum(hit.values()) / total, move_exact=sum(moved.values()) / total,
                   **{k: hit[k] / n[k] for k in sorted(n)})
        out[name] = row
        print(a.arm, name, {k: round(v, 3) for k, v in row.items()}, f"{time.time() - t:.0f}s", flush=True)
        json.dump(log, open(f"{HERE}/data/out_{a.arm}_{name}.json", "w"))
    json.dump(out, open(f"{HERE}/data/score_{a.arm}.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--max-tokens", type=int, default=40)
    main(p.parse_args())
