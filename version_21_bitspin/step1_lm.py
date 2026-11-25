"""step1_lm.py — how well does a bits-and-counters model predict English?

Same protocol as version_19 step2, so the numbers line up: learn from five
Gutenberg novels (4.0 MB), then predict the first ~200k chars of *Pride and
Prejudice*, held out. Reports bits/char two ways:

  adapting : the model keeps counting while it reads the test (as v19 did)
  frozen   : counters locked after training

v19's order-5 n-gram scored 1.900 adapting on the same text.

Run:  python step1_lm.py [--chars 200000] [--train-mb 0] [--ablate]
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "version_19_glyph"))
from step2_codec import TRAIN, book  # noqa: E402

import bitspin  # noqa: E402


def corpus(train_mb=0):
    data = "\n\n".join(book(k, v) for k, v in TRAIN.items()).encode("utf-8")
    return data[: int(train_mb * 1e6)] if train_mb else data


def test_text(chars):
    text = book("pride_and_prejudice", 1342)[:chars]
    return text[: text.rfind("\n\n")].encode("utf-8")


def score(heads, train, test, label):
    t0 = time.time()
    m = bitspin.BitSpin(heads)
    m.run(train)
    tt = time.time() - t0
    snap = [h.n.copy() for h in m.heads], m.logw.copy()
    frozen = m.run(test, learn=False, tail=train) / len(test)
    for h, n in zip(m.heads, snap[0]):
        h.n = n
    m.logw = snap[1]
    adapt = m.run(test, learn=True, batch=256, tail=train) / len(test)
    print(f"{label:<28} adapting {adapt:.3f}  frozen {frozen:.3f}  "
          f"({m.memory_bytes() / 1e6:.0f} MB counters, train {tt:.0f}s)", flush=True)
    return adapt, frozen


def main(chars, train_mb, ablate):
    train, test = corpus(train_mb), test_text(chars)
    print(f"train {len(train) / 1e6:.2f} MB, test {len(test):,} chars (Pride and Prejudice)\n")
    score(bitspin.DEFAULT_HEADS, train, test, "bitspin (all heads)")
    if ablate:
        H = bitspin.DEFAULT_HEADS
        score([h for h in H if h[0] != "soft"], train, test, "without soft (angle) heads")
        score([h for h in H if h[0] != "word"], train, test, "without word heads")
        score([h for h in H if h[0] == "exact"], train, test, "byte contexts only")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=200_000)
    ap.add_argument("--train-mb", type=float, default=0)
    ap.add_argument("--ablate", action="store_true")
    a = ap.parse_args()
    main(a.chars, a.train_mb, a.ablate)
