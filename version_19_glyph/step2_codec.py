"""step2_codec.py — compress English into pixels and read it back exactly.

Train the glyph language's prior on five Gutenberg novels, then write a
held-out novel (Pride and Prejudice) into an RGB canvas two ways:

  stream : one canvas, one context (best density)
  tiles  : every paragraph its own self-delimiting run (random access)

Each canvas is saved as PNG, read back from disk, decoded and compared
byte-for-byte. Reports bits/char and chars/pixel against zlib/bz2/lzma and
a typeset-font canvas (what DeepSeek-OCR / Glyph feed their encoders).

Run:  python step2_codec.py [--chars 200000] [--order 5]
"""

import argparse
import bz2
import lzma
import os
import time
import urllib.request
import zlib

import glyph

TRAIN = {"moby_dick": 2701, "sherlock": 1661, "frankenstein": 84,
         "two_cities": 98, "great_expectations": 1400}
TEST = {"pride_and_prejudice": 1342}
FONT_PX_PER_CHAR = 8 * 16 / 3   # 8x16 mono cell, 1 bit per subpixel, packed RGB

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def book(name, n):
    path = os.path.join(DATA, f"{name}.txt")
    if not os.path.exists(path):
        os.makedirs(DATA, exist_ok=True)
        url = f"https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt"
        urllib.request.urlretrieve(url, path)
    text = open(path, encoding="utf-8-sig").read().replace("\r\n", "\n")
    a = text.find("\n", text.find("*** START OF")) + 1
    b = text.find("*** END OF")
    return text[a:b].strip()


def paragraphs(text):
    out = []
    for p in (q.strip() for q in text.split("\n\n")):
        while p:
            cut = p[:glyph.TILE_MAX // 2]
            out.append(cut)
            p = p[len(cut):]
    return out


def main(chars, order):
    t0 = time.time()
    model = glyph.Model(order)
    corpus = "\n\n".join(book(k, v) for k, v in TRAIN.items()).encode("utf-8")
    model.train(corpus)
    print(f"prior: order-{order} on {len(corpus) / 1e6:.2f} MB of Gutenberg "
          f"({time.time() - t0:.0f}s)")

    text = book("pride_and_prejudice", TEST["pride_and_prejudice"])[:chars]
    text = text[:text.rfind("\n\n")]          # end on a paragraph
    raw = text.encode("utf-8")
    n = len(text)
    print(f"test:  Pride and Prejudice, held out, {n:,} chars\n")

    rows = []

    t = time.time()
    px = glyph.encode_stream(model, text)
    path = os.path.join(HERE, "glyph_stream.png")
    glyph.write_png(path, glyph.to_canvas(px))
    back = glyph.decode_stream(model, glyph.from_canvas(glyph.read_png(path)))
    side = glyph.read_png(path).shape[0]
    rows.append(("glyph stream", len(px) * 24, len(px), back == text,
                 f"{side}x{side} canvas, {time.time() - t:.0f}s"))

    t = time.time()
    paras = paragraphs(text)
    px = glyph.encode_tiles(model, paras)
    path = os.path.join(HERE, "glyph_tiles.png")
    glyph.write_png(path, glyph.to_canvas(px))
    back = glyph.decode_tiles(model, glyph.from_canvas(glyph.read_png(path)))
    side = glyph.read_png(path).shape[0]
    rows.append(("glyph tiles", len(px) * 24, len(px), back == paras,
                 f"{len(paras):,} paragraphs, {side}x{side}, {time.time() - t:.0f}s"))

    for name, fn in (("zlib -9", lambda b: zlib.compress(b, 9)),
                     ("bz2 -9", lambda b: bz2.compress(b, 9)),
                     ("lzma -9e", lambda b: lzma.compress(b, preset=9 | lzma.PRESET_EXTREME))):
        bits = len(fn(raw)) * 8
        rows.append((name, bits, bits / 24, None, "same bits packed into RGB"))
    rows.append(("typeset font", None, n * FONT_PX_PER_CHAR, None,
                 "8x16 mono, 1 bit/subpixel"))

    print(f"{'codec':<14}{'bits/char':>10}{'chars/px':>10}{'exact':>7}   note")
    for name, bits, npx, ok, note in rows:
        bpc = f"{bits / n:.3f}" if bits else "-"
        ex = "-" if ok is None else ("yes" if ok else "NO")
        print(f"{name:<14}{bpc:>10}{n / npx:>10.2f}{ex:>7}   {note}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=200_000)
    ap.add_argument("--order", type=int, default=glyph.ORDER)
    a = ap.parse_args()
    main(a.chars, a.order)
