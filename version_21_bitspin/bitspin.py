"""bitspin.py — a language model made of bits and counters.

Text goes in as raw bits and comes out one bit at a time. There are no
real-valued weights and no backprop. Every forward pass is also the
training step: after each bit, the counters behind it move by one.

The pieces, in the angle language of the idea (Chris, 2026-11-08):

  byte code   every byte value is a 1024-bit vector built from its own 8
              bits: XOR of a fixed random vector per set bit (bit = 0 or π)
  rotation    position is rotation: a byte k steps back is rolled k words,
              so the same byte at another distance is a different angle
  binding     XOR of rotated bytes = one exact context (any change -> 90°)
  bundling    majority of rotated bytes = a soft context; contexts that
              share bytes sit at a small angle to each other
  heads       each head turns a context into an address (exact: XOR-bound;
              soft: k sampled bits of the bundle, so two contexts collide
              with probability (1 - angle/π)^k)
  counters    each address holds two counters (n0, n1) for the next bit;
              the bit the head predicts is their sign, the size is its
              confidence; learning = +1 on the bit that came
  mixing      heads are combined by weights per (head, confidence level),
              moved multiplicatively by how well each head just did

Batches: B bytes are predicted with the counters as they stood at the
start of the batch, then all their bits are counted in. B is the only lag.
"""

import array

import numpy as np

WORDY = ("word", "memory")
D_WORDS = 16                    # 16 x 64 = 1024-bit vectors
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def mix64(x):
    """splitmix64 finaliser, vectorised (uint64 wraps)."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def byte_codes(seed=0):
    """HV[b] (256 x 16 uint64): XOR of one random vector per set bit of b."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 2**63, D_WORDS, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    per_bit = rng.integers(0, 2**63, (8, D_WORDS), dtype=np.uint64) * np.uint64(2) + \
        rng.integers(0, 2, (8, D_WORDS), dtype=np.uint64)
    hv = np.tile(base, (256, 1))
    for b in range(256):
        for i in range(8):
            if (b >> i) & 1:
                hv[b] ^= per_bit[i]
    return hv


def unpack(words):
    """(..., 16) uint64 -> (..., 1024) uint8 bits."""
    return np.unpackbits(words.view(np.uint8), axis=-1, bitorder="little")


def pack(bits):
    """(..., 1024) bits -> (..., 16) uint64."""
    return np.packbits(bits.astype(np.uint8), axis=-1, bitorder="little").view(np.uint64)


def state_class(n):
    """Trust level read off a counter pair itself, not off a shared table.

    0..6  : never contradicted, seen 0..6+ times
    7..11 : both bits seen, by total count (log2)
    """
    lo, tot = np.minimum(n[..., 0], n[..., 1]), n[..., 0] + n[..., 1]
    clean = np.minimum(tot, 6).astype(np.int64)
    mixed = 7 + np.minimum(np.log2(np.maximum(tot, 1)).astype(np.int64) - 1, 4)
    return np.where(lo == 0, clean, mixed)


def word_state(data, maxwords):
    """Per byte position t, from data[:t]: [partial word, word 1 back, ... ].

    Words are runs of letters, lowercased, each hashed to 64 bits.
    """
    out = np.zeros((len(data), maxwords + 1), np.uint64)
    if not maxwords:
        return out
    prev = [0] * maxwords
    cur = 0
    mask = (1 << 64) - 1
    # one row per word boundary, then each position points at its row
    rows, at = [[0] * maxwords], array.array("q", bytes(8 * len(data)))
    part = array.array("Q", bytes(8 * len(data)))
    for t, c in enumerate(data):
        part[t] = cur
        at[t] = len(rows) - 1
        if 65 <= c <= 90 or 97 <= c <= 122:
            cur = ((cur ^ (c | 32)) * 0x100000001B3 + 1) & mask
        elif cur:
            prev = [cur] + prev[:-1]
            rows.append(prev)
            cur = 0
    out[:, 0] = np.frombuffer(part, np.uint64)
    out[:, 1:] = np.array(rows, dtype=np.uint64)[np.frombuffer(at, np.int64)]
    return out


class Head:
    """One way to read the context into an address.

    kind 'exact': XOR-bind the rotated bytes at distances `lags`.
    kind 'soft':  majority-bundle them, then sample `k` bits as the address.
    """

    def __init__(self, kind, lags, k=0, bits=22, seed=0):
        self.kind, self.lags, self.k = kind, tuple(lags), k
        self.size = 1 << bits
        rng = np.random.default_rng(1000 + seed)
        self.salt = np.uint64(rng.integers(1, 2**63))
        if kind == "soft":
            self.sample = rng.choice(64 * D_WORDS, k, replace=False)
            self.tie = rng.integers(0, 2, 64 * D_WORDS, dtype=np.uint8)
        self.n = np.zeros((self.size, 2), np.uint16)
        self.tag = np.zeros(self.size, np.uint8)

    def name(self):
        lag = ",".join(map(str, self.lags)) if len(self.lags) < 5 else f"1..{max(self.lags)}"
        return f"{self.kind}[{lag}]" + (f"k{self.k}" if self.kind == "soft" else "")

    def keys(self, hv, hist, t, words=None):
        """64-bit context key for byte positions t (hist = padded byte array)."""
        if self.kind in WORDY:
            w = words[t]
            key = np.full(len(t), self.salt, np.uint64) ^ w[:, 0]
            for lag in self.lags:              # word k back = its code turned 7k bits
                r = np.uint64((7 * lag) % 64)
                key ^= (w[:, lag] << r) | (w[:, lag] >> (np.uint64(64) - r))
            return mix64(key)
        if self.kind == "exact":
            key = np.full(len(t), self.salt, np.uint64)
            for lag in self.lags:
                key ^= hv[hist[t - lag], lag % D_WORDS] * np.uint64(2 * lag + 1)
            return mix64(key)
        acc = np.zeros((len(t), 64 * D_WORDS), np.int16)
        for lag in self.lags:
            acc += unpack(np.roll(hv[hist[t - lag]], lag, axis=1))
        two = 2 * acc
        m = len(self.lags)
        bits = (two > m) | ((two == m) & self.tie.astype(bool))
        s = bits[:, self.sample].astype(np.uint64)
        key = np.zeros(len(t), np.uint64)
        for j in range(self.k):
            key |= s[:, j] << np.uint64(j)
        return mix64(key ^ self.salt)

    def slots(self, key, prefix):
        """Address for (context, bits-so-far-of-this-byte)."""
        x = mix64(key[:, None] ^ (prefix.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)))
        return ((x & np.uint64(self.size - 1)).astype(np.int64),
                (x >> np.uint64(56)).astype(np.uint8))

    def read(self, idx, tag):
        """Counters at idx; a slot owned by another context reads as empty."""
        n = self.n[idx].astype(np.float64)
        n[self.tag[idx] != tag] = 0
        return n

    def write(self, idx, tag, y, limit, fade=False):
        stale = self.tag[idx] != tag
        if stale.any():                       # newest context takes the slot
            self.n[idx[stale]] = 0
            self.tag[idx[stale]] = tag[stale]
        if fade:                              # a correction pushes the old bit's count back
            flat_opp = (idx * 2 + 1 - y).ravel()
            v = self.n.reshape(-1)[flat_opp]
            self.n.reshape(-1)[flat_opp] = v // 2 if fade == 2 else np.where(v > 2, v // 2 + 1, v)
        np.add.at(self.n.reshape(-1), (idx * 2 + y).ravel(), 1)
        u = np.unique(idx)
        over = u[self.n[u].sum(1, dtype=np.int64) > limit]
        if len(over):
            self.n[over] = (self.n[over] + 1) // 2


DEFAULT_HEADS = [
    ("exact", [1]), ("exact", [1, 2]), ("exact", [1, 2, 3]),
    ("exact", [1, 2, 3, 4]), ("exact", range(1, 6)), ("exact", range(1, 7)),
    ("exact", range(1, 9)),
    ("soft", range(1, 8), 20), ("soft", range(1, 13), 24),
    ("word", [1]), ("word", [1, 2]), ("word", [1, 2, 3]), ("memory", range(1, 7)),
]


class BitSpin:
    def __init__(self, heads=DEFAULT_HEADS, bits=22, conf_levels=12, limit=1023,
                 eta=0.02, prior=0.4, beta=8.0, fade="memory", beta_word=1.0, seed=0):
        self.hv = byte_codes(seed)
        self.heads = [Head(h[0], h[1], h[2] if len(h) > 2 else 0, bits, seed + i)
                      for i, h in enumerate(heads)]
        self.maxlag = max(max(h.lags) for h in self.heads if h.kind not in WORDY)
        self.maxwords = max([max(h.lags) for h in self.heads if h.kind in WORDY] or [0])
        self.levels, self.limit, self.eta, self.prior = conf_levels, limit, eta, prior
        self.beta, self.fade, self.beta_word = beta, fade, beta_word
        self.mix_learns = True       # False: only counters move (the mix is shared, counters are local)
        n_soft = sum(h.kind in ("soft", "word") for h in self.heads)
        self.logw = np.zeros((1 + n_soft, conf_levels))

    def fades(self, h):
        """fade = True (every head, gently), False, or 'memory' (only the
        memory head, hard: there one context is one fact)."""
        if self.fade == "memory":
            return 2 if h.kind == "memory" else 0
        return 1 if self.fade else 0

    def memory_bytes(self):
        return sum(h.n.nbytes + h.tag.nbytes for h in self.heads)

    def run(self, data, learn=True, batch=2048, tail=b""):
        """Predict every bit of `data`; return total bits of code (-log2 p).

        `tail` is earlier text used only as context for the first bytes.
        """
        pad = self.maxlag
        ctx = tail[-4096:].rjust(pad, b"\0")
        hist = np.frombuffer(ctx[-pad:] + data, np.uint8).astype(np.int64)
        start = pad
        words = None
        if self.maxwords:
            words = word_state(ctx + data, self.maxwords)[len(ctx) - pad:]
        total, per_byte = 0.0, np.empty(len(data))
        for b0 in range(0, len(data), batch):
            t = np.arange(start + b0, start + min(b0 + batch, len(data)))
            c = hist[t]
            i = np.arange(8)
            y = ((c[:, None] >> (7 - i)) & 1).astype(np.int64)            # (B, 8)
            prefix = (1 << i) | (c[:, None] >> (8 - i))                    # (B, 8)
            prefix[:, 0] = 1
            # exact heads form a back-off chain, short contexts first: each
            # longer context's counters refine the shorter one's guess
            # (Witten-Bell at bit level: trust grows with count, shrinks
            # with how many different bits that context has seen)
            ps, cls, idxs, words_n = [], [], [], []
            chain = np.full(y.shape, 0.5)
            depth = np.zeros(y.shape, np.int64)
            for j, h in enumerate(self.heads):
                idx, tag = h.slots(h.keys(self.hv, hist, t, words), prefix)  # (B, 8)
                n = h.read(idx, tag)                                       # (B, 8, 2)
                tot = n[..., 0] + n[..., 1]
                idxs.append((idx, tag))
                if h.kind == "exact":
                    beta = self.beta * ((n[..., 0] > 0).astype(float) + (n[..., 1] > 0))
                    seen = tot > 0
                    chain = np.where(seen, (n[..., 1] + beta * chain) / (tot + beta + 1e-12), chain)
                    depth = np.where(seen, j + 1, depth)
                elif h.kind == "memory":
                    words_n.append(n)
                else:
                    ps.append((n[..., 1] + self.prior) / (tot + 2 * self.prior))
                    cls.append(state_class(n))
            ps.insert(0, chain)
            cls.insert(0, np.minimum(depth, self.levels - 1))
            P = np.stack(ps)                                               # (K, B, 8)
            C = np.stack(cls)
            hidx = np.arange(len(P))[:, None, None]
            W = np.exp(self.logw[hidx, C])
            pmix = np.clip((W * P).sum(0) / W.sum(0), 1e-6, 1 - 1e-6)
            # word heads are memory: they refine the final guess by their own
            # counts alone (no shared weight), shortest first
            p1 = pmix
            for n in words_n:
                tot = n[..., 0] + n[..., 1]
                p1 = np.where(tot > 0, (n[..., 1] + self.beta_word * p1) / (tot + self.beta_word), p1)
            p1 = np.clip(p1, 1e-6, 1 - 1e-6)
            py = np.where(y == 1, p1, 1 - p1)
            pmix_y = np.where(y == 1, pmix, 1 - pmix)
            bits = -np.log2(py)
            per_byte[b0:b0 + len(t)] = bits.sum(1)
            total += bits.sum()
            if not learn:
                continue
            if not self.mix_learns:
                for h, (idx, tag) in zip(self.heads, idxs):
                    h.write(idx, tag, y, self.limit, self.fades(h))
                continue
            # mixing weights: each head gets credit relative to the mix
            ph = np.where(y == 1, P, 1 - P)
            gain = np.log(np.clip(ph, 1e-6, 1) / pmix_y)                       # (H, B, 8)
            g = np.zeros_like(self.logw)
            cnt = np.zeros_like(self.logw)
            np.add.at(g, (np.broadcast_to(hidx, C.shape), C), gain)
            np.add.at(cnt, (np.broadcast_to(hidx, C.shape), C), 1)
            self.logw = np.clip(self.logw + self.eta * g / np.maximum(cnt, 1) * np.minimum(cnt, 50),
                                -12, 12)
            # counters: +1 on the bit that came
            for h, (idx, tag) in zip(self.heads, idxs):
                h.write(idx, tag, y, self.limit, self.fades(h))
        self.last_per_byte = per_byte
        return total
