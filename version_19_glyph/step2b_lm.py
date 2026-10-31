"""step2b_lm.py — the glyph language with Qwen3-0.6B as its prior.

Same held-out Pride and Prejudice text as step2, same PNG round-trip, now
with a language model predicting each token. The order-5 n-gram prior runs on
the identical text alongside it for a like-for-like comparison.

Run:  .venv/bin/python step2b_lm.py [--chars 50000]
"""

import argparse
import glob
import os
import time

import glyph
import glyph_lm
from step2_codec import HERE, TEST, TRAIN, book, paragraphs


def unseen():
    """The thinktank's own write-ups: all written in 2026, after Qwen3 trained."""
    root = os.path.dirname(HERE)
    files = sorted(glob.glob(f"{root}/version_*/*.md") + glob.glob(f"{root}/frontier/*.md")
                   + [f"{root}/paper/README.md"])
    return "\n\n".join(open(f).read().strip() for f in files if "version_19" not in f)


def main(chars, source):
    text = (unseen() if source == "unseen"
            else book("pride_and_prejudice", TEST["pride_and_prejudice"]))[:chars]
    text = text[:text.rfind("\n\n")]
    paras = paragraphs(text)
    n = len(text)
    rows = []

    t = time.time()
    ng = glyph.Model(5)
    ng.train("\n\n".join(book(k, v) for k, v in TRAIN.items()).encode("utf-8"))
    px = glyph.encode_stream(ng, text)
    ok = glyph.decode_stream(ng, px) == text
    rows.append(("n-gram stream", len(px), ok, f"{time.time() - t:.0f}s"))

    lm = glyph_lm.LMPrior()
    label = "thinktank write-ups (2026, unseen)" if source == "unseen" else "Pride and Prejudice"
    print(f"test: {label}, {n:,} chars, "
          f"{len(lm.tokens(text)):,} Qwen tokens\n")

    for mode, enc, dec, src in (
            ("lm stream", glyph_lm.encode_stream, glyph_lm.decode_stream, text),
            ("lm tiles", glyph_lm.encode_tiles, glyph_lm.decode_tiles, paras)):
        t = time.time()
        px = enc(lm, src)
        path = os.path.join(HERE, f"glyph_{source}_{mode.replace(' ', '_')}.png")
        glyph.write_png(path, glyph.to_canvas(px))
        img = glyph.read_png(path)
        te = time.time() - t
        ok = dec(lm, glyph.from_canvas(img)) == src
        side = img.shape[0]
        rows.append((mode, len(px), ok,
                     f"{side}x{side}, encode {te:.0f}s, decode {time.time() - t - te:.0f}s"))

    print(f"{'prior':<15}{'bits/char':>10}{'chars/px':>10}{'exact':>7}   note")
    for name, npx, ok, note in rows:
        print(f"{name:<15}{npx * 24 / n:>10.3f}{n / npx:>10.2f}"
              f"{'yes' if ok else 'NO':>7}   {note}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=50_000)
    ap.add_argument("--source", choices=["pp", "unseen"], default="pp")
    a = ap.parse_args()
    main(a.chars, a.source)
