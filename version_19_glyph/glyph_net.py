"""glyph_net.py — a learned glyph code that is read in one parallel pass.

A writer network turns a chunk of L characters into a P×P RGB patch; a reader
network turns the patch back into all L characters at once. No arithmetic
coding, no left-to-right decode: the whole patch is read in a single forward
pass. Between them the patch goes through a real 8-bit channel (uniform
quantization noise while training, true rounding at eval), plus optional
extra noise so the code only uses bits that survive.
"""

import string

import mlx.core as mx
import mlx.nn as nn
import numpy as np

VOCAB = sorted(set(string.printable) - set("\t\r\x0b\x0c"))
V = len(VOCAB) + 1           # last id = anything else
_LUT = np.full(256, V - 1, np.int32)
for i, ch in enumerate(VOCAB):
    _LUT[ord(ch)] = i


def ids(text):
    b = np.frombuffer(text.encode("ascii", "replace"), np.uint8)
    return _LUT[b]


def text(idv):
    return "".join(VOCAB[i] if i < len(VOCAB) else "?" for i in idv)


class Writer(nn.Module):
    def __init__(self, L, P, d=256, layers=4, heads=8):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = mx.random.normal((L, d)) * 0.02
        self.enc = nn.TransformerEncoder(layers, d, heads)
        self.out = nn.Linear(L * d, P * P * 3)
        self.P = P

    def __call__(self, x):                       # (B, L) -> (B, P, P, 3) in [0,1]
        h = self.enc(self.emb(x) + self.pos, None)
        return mx.sigmoid(self.out(h.reshape(h.shape[0], -1))).reshape(-1, self.P, self.P, 3)


class Reader(nn.Module):
    def __init__(self, L, P, d=256, layers=4, heads=8):
        super().__init__()
        self.inp = nn.Linear(P * P * 3, L * d)
        self.pos = mx.random.normal((L, d)) * 0.02
        self.enc = nn.TransformerEncoder(layers, d, heads)
        self.out = nn.Linear(d, V)
        self.L, self.d = L, d

    def __call__(self, img):                     # (B, P, P, 3) -> (B, L, V)
        h = self.inp(img.reshape(img.shape[0], -1)).reshape(-1, self.L, self.d)
        return self.out(self.enc(h + self.pos, None))


def channel(img, sigma, train):
    """8-bit pixels, plus Gaussian noise of sigma (in 0-255 units)."""
    if train:
        img = img + (mx.random.uniform(shape=img.shape) - 0.5) / 255
    else:
        img = mx.round(img * 255) / 255
    if sigma:
        img = img + mx.random.normal(img.shape) * sigma / 255
    return img


def save(path, writer, reader):
    writer.save_weights(path + ".writer.safetensors")
    reader.save_weights(path + ".reader.safetensors")


def load(path, L, P):
    w, r = Writer(L, P), Reader(L, P)
    w.load_weights(path + ".writer.safetensors")
    r.load_weights(path + ".reader.safetensors")
    return w, r
