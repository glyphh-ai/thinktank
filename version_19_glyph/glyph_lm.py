"""glyph_lm.py — the glyph language with a real language model as its prior.

Same coder and pixels as glyph.py, but the probability of each next token
comes from Qwen3-0.6B (MLX, 4-bit) instead of the n-gram tables. Encoder and
decoder run the model on exactly the same path (same prime, then one token at
a time through the KV cache), so the logits match bit for bit and the text
comes back exact.

Long text is coded in windows of W tokens; each window restarts the cache
primed with the previous P tokens, so context carries over the seam.
"""

import glob
import os

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

import glyph

MODEL = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/*"))[0]
SCALE = 1 << 24


class LMPrior:
    def __init__(self, path=MODEL, window=1024, prime=128):
        self.model, self.tok = load(path)
        self.start = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.window, self.prime = window, prime

    def _cum(self, logits):
        p = np.array(mx.softmax(logits.astype(mx.float32)), np.float64)
        f = (p * (SCALE - len(p))).astype(np.int64) + 1
        cum = np.zeros(len(p) + 1, np.int64)
        np.cumsum(f, out=cum[1:])
        return cum

    def _run(self, n, choose):
        """Shared encode/decode loop; choose(cum) picks or reads each token."""
        toks = []
        for i in range(n):
            if i % self.window == 0:
                cache = make_prompt_cache(self.model)
                ctx = [self.start] + toks[max(0, i - self.prime):i]
                logits = self.model(mx.array(ctx)[None], cache=cache)[0, -1]
            t = choose(self._cum(logits))
            toks.append(t)
            logits = self.model(mx.array([[t]]), cache=cache)[0, -1]
        return toks

    def tokens(self, text):
        ids = self.tok.encode(text, add_special_tokens=False)
        assert self.tok.decode(ids) == text, "tokenizer does not round-trip"
        return ids

    def encode_bits(self, text):
        ids = self.tokens(text)
        enc, it = glyph._Encoder(), iter(ids)

        def put(cum):
            t = next(it)
            enc.put(cum, t)
            return t

        self._run(len(ids), put)
        return len(ids), enc.finish()

    def decode_bits(self, bits, ntok):
        dec = glyph._Decoder(bits)
        return self.tok.decode(self._run(ntok, dec.get))


def encode_stream(lm, text):
    n, bits = lm.encode_bits(text)
    return np.vstack([glyph._u24(n), glyph.bits_to_pixels(bits)])


def decode_stream(lm, px):
    return lm.decode_bits(glyph.pixels_to_bits(px[1:]), glyph._from24(px[0]))


# Tiles: header pixel = 13-bit token count | 11-bit pixel count.
def encode_tiles(lm, passages):
    out = []
    for t in passages:
        n, bits = lm.encode_bits(t)
        px = glyph.bits_to_pixels(bits)
        assert 0 < n < 8192 and len(px) < 2048
        out += [glyph._u24(n << 11 | len(px)), px]
    return np.vstack(out)


def decode_tiles(lm, px):
    out, i = [], 0
    while i < len(px):
        h = glyph._from24(px[i])
        n, m = h >> 11, h & 2047
        if not n:
            break
        out.append(lm.decode_bits(glyph.pixels_to_bits(px[i + 1:i + 1 + m]), n))
        i += 1 + m
    return out
