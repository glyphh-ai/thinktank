"""glyph.py — the glyph language: English -> pixels, losslessly.

A predictive model of English gives every next byte a probability; an
arithmetic coder turns those probabilities into bits (about -log2 p each);
the bits fill RGB pixels 24 at a time. Decoding runs the same model in
lockstep and reads the pixels back into the exact same text.

The language *is* the model: the better it predicts English, the more
characters fit in a pixel. Prior here is a numpy order-N byte model
(Witten-Bell blended), trained on a corpus and adapting as it reads.
"""

import struct
import zlib

import numpy as np

ORDER = 5
SCALE = 1 << 16          # integer frequency total per step
TOP, HALF, QTR = (1 << 32) - 1, 1 << 31, 1 << 30


# ---------------------------------------------------------------- the model

class Model:
    """Blended order-0..N byte contexts. tables[k][ctx] = {byte: count}."""

    def __init__(self, order=ORDER):
        self.order = order
        self.tables = [dict() for _ in range(order + 1)]

    def update(self, hist, c, log=None):
        for k in range(min(self.order, len(hist)) + 1):
            ctx = bytes(hist[len(hist) - k:]) if k else b""
            t = self.tables[k].get(ctx)
            if t is None:
                t = self.tables[k][ctx] = {}
            t[c] = t.get(c, 0) + 1
            if log is not None:
                log.append((k, ctx, c))

    def revert(self, log):
        """Undo a session's updates so the model returns to its trained state."""
        for k, ctx, c in reversed(log):
            t = self.tables[k][ctx]
            t[c] -= 1
            if not t[c]:
                del t[c]
                if not t:
                    del self.tables[k][ctx]

    def train(self, data):
        n = self.order
        for i in range(len(data)):
            self.update(data[max(0, i - n):i], data[i])

    def dist(self, hist):
        """P(next byte | hist) as a 256-vector, low orders blended into high."""
        p = np.full(256, 1 / 256)
        for k in range(min(self.order, len(hist)) + 1):
            t = self.tables[k].get(bytes(hist[len(hist) - k:]) if k else b"")
            if not t:
                break    # an unseen context has no seen extensions either
            idx = np.fromiter(t.keys(), np.int64, len(t))
            cnt = np.fromiter(t.values(), np.float64, len(t))
            z = cnt.sum() + len(t)
            p *= len(t) / z
            p[idx] += cnt / z
        return p


def _cum(p):
    f = (p * (SCALE - 256)).astype(np.int64) + 1     # every byte stays codable
    cum = np.zeros(257, np.int64)
    np.cumsum(f, out=cum[1:])
    return cum


# ------------------------------------------------------- arithmetic coding

class _Encoder:
    def __init__(self):
        self.low, self.high, self.pending, self.bits = 0, TOP, 0, []

    def _emit(self, b):
        self.bits.append(b)
        self.bits.extend([1 - b] * self.pending)
        self.pending = 0

    def put(self, cum, s):
        r, tot = self.high - self.low + 1, int(cum[-1])
        self.high = self.low + r * int(cum[s + 1]) // tot - 1
        self.low = self.low + r * int(cum[s]) // tot
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QTR and self.high < 3 * QTR:
                self.pending += 1
                self.low -= QTR
                self.high -= QTR
            else:
                break
            self.low, self.high = 2 * self.low, 2 * self.high + 1

    def finish(self):
        self.pending += 1
        self._emit(0 if self.low < QTR else 1)
        return self.bits


class _Decoder:
    def __init__(self, bits):
        self.bits, self.i, self.low, self.high, self.code = bits, 0, 0, TOP, 0
        for _ in range(32):
            self.code = 2 * self.code + self._next()

    def _next(self):
        b = self.bits[self.i] if self.i < len(self.bits) else 0
        self.i += 1
        return b

    def get(self, cum):
        r, tot = self.high - self.low + 1, int(cum[-1])
        v = ((self.code - self.low + 1) * tot - 1) // r
        s = int(np.searchsorted(cum, v, side="right")) - 1
        self.high = self.low + r * int(cum[s + 1]) // tot - 1
        self.low = self.low + r * int(cum[s]) // tot
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QTR and self.high < 3 * QTR:
                self.low -= QTR
                self.high -= QTR
                self.code -= QTR
            else:
                break
            self.low, self.high = 2 * self.low, 2 * self.high + 1
            self.code = 2 * self.code + self._next()
        return s


def encode_bits(model, data, adapt=True):
    """bytes -> list of bits. Leaves the model exactly as it found it."""
    enc, log, n = _Encoder(), [], model.order
    for i, c in enumerate(data):
        hist = data[max(0, i - n):i]
        enc.put(_cum(model.dist(hist)), c)
        if adapt:
            model.update(hist, c, log)
    model.revert(log)
    return enc.finish()


def decode_bits(model, bits, nbytes, adapt=True):
    dec, log, n, out = _Decoder(bits), [], model.order, bytearray()
    for i in range(nbytes):
        hist = out[max(0, i - n):i]
        c = dec.get(_cum(model.dist(hist)))
        out.append(c)
        if adapt:
            model.update(hist, c, log)
    model.revert(log)
    return bytes(out)


# ----------------------------------------------------------- bits <-> pixels

def bits_to_pixels(bits):
    b = np.packbits(np.array(bits, np.uint8))
    b = np.concatenate([b, np.zeros(-len(b) % 3, np.uint8)])
    return b.reshape(-1, 3)


def pixels_to_bits(px):
    return np.unpackbits(px.reshape(-1).astype(np.uint8)).tolist()


def _u24(n):
    return np.array([[n >> 16 & 255, n >> 8 & 255, n & 255]], np.uint8)


def _from24(px):
    return int(px[0]) << 16 | int(px[1]) << 8 | int(px[2])


# Stream: one canvas, one context. Header = 2 pixels (48-bit byte count).
def encode_stream(model, text):
    data = text.encode("utf-8")
    px = bits_to_pixels(encode_bits(model, data))
    return np.vstack([_u24(len(data) >> 24), _u24(len(data) & 0xFFFFFF), px])


def decode_stream(model, px):
    n = _from24(px[0]) << 24 | _from24(px[1])
    return decode_bits(model, pixels_to_bits(px[2:]), n).decode("utf-8")


# Tiles: each passage coded on its own, so any one can be read without the
# rest. Header pixel = 13-bit byte count | 11-bit pixel count, self-delimiting.
TILE_MAX = 6000

def encode_tiles(model, passages):
    out = []
    for t in passages:
        data = t.encode("utf-8")
        assert 0 < len(data) <= TILE_MAX, len(data)
        px = bits_to_pixels(encode_bits(model, data))
        assert len(px) < 2048
        out += [_u24(len(data) << 11 | len(px)), px]
    return np.vstack(out)


def decode_tiles(model, px):
    out, i = [], 0
    while i < len(px):
        h = _from24(px[i])
        n, m = h >> 11, h & 2047
        if not n:
            break        # canvas padding
        out.append(decode_bits(model, pixels_to_bits(px[i + 1:i + 1 + m]), n).decode("utf-8"))
        i += 1 + m
    return out


# ------------------------------------------------------------- the canvas

def to_canvas(px):
    side = int(np.ceil(np.sqrt(len(px))))
    img = np.zeros((side * side, 3), np.uint8)
    img[:len(px)] = px
    return img.reshape(side, side, 3)


def from_canvas(img):
    return img.reshape(-1, 3)


def write_png(path, img):
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, body):
        return (struct.pack(">I", len(body)) + tag + body
                + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9))
                + chunk(b"IEND", b""))


def read_png(path):
    """Reads the PNGs write_png makes (8-bit RGB, filter 0)."""
    with open(path, "rb") as f:
        buf = f.read()
    i, idat = 8, b""
    while i < len(buf):
        n, tag = struct.unpack(">I4s", buf[i:i + 8])
        body = buf[i + 8:i + 8 + n]
        if tag == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            idat += body
        i += 12 + n
    rows = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + 3 * w)
    assert not rows[:, 0].any(), "only filter-0 PNGs"
    return rows[:, 1:].reshape(h, w, 3).copy()
