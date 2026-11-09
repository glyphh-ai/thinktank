"""memory.py — counter memories that sit beside Qwen and learn on every pass.

Both refine Qwen's next-token probability by their own counts, the rule
version_21's memory head used:

    p(tok) = (count(tok) + beta * p_qwen(tok)) / (N + beta)

so with nothing stored Qwen is untouched, and a fact lives only in its
own counters. A contradiction halves the old token's count (1 -> 0).

ExactMemory  key = the last n tokens, XOR-bound (n = 4, 8, 12, short first)
AngleMemory  key = the sign bits of Qwen's hidden state at one layer
             (2048 dims -> 2048 bits, each 0 or π after centering); a stored
             entry fires when its Hamming distance (XOR + popcount, i.e. the
             angle between the two states) is within tau
"""

import numpy as np


class ExactMemory:
    def __init__(self, orders=(4, 8, 12), beta=1.0):
        self.orders, self.beta = orders, beta
        self.tables = {n: {} for n in orders}

    def _keys(self, tokens, t):
        """keys for predicting tokens[t + 1] (context = tokens[..t])."""
        for n in self.orders:
            if t + 1 >= n:
                yield n, hash(tuple(tokens[t + 1 - n:t + 1]))

    def learn(self, tokens):
        for t in range(len(tokens) - 1):
            nxt = tokens[t + 1]
            for n, key in self._keys(tokens, t):
                c = self.tables[n].setdefault(key, {})
                for other in [o for o in c if o != nxt]:
                    c[other] //= 2
                    if not c[other]:
                        del c[other]
                c[nxt] = c.get(nxt, 0) + 1

    def refine(self, tokens, lp, start=0):
        """lp[t] = log p(tokens[t+1]); returns refined copy (positions >= start)."""
        lp = lp.copy()
        for t in range(start, len(tokens) - 1):
            p = np.exp(lp[t])
            for n, key in self._keys(tokens, t):
                c = self.tables[n].get(key)
                if c:
                    p = (c.get(tokens[t + 1], 0) + self.beta * p) / (sum(c.values()) + self.beta)
            lp[t] = np.log(max(p, 1e-30))
        return lp


def codes(h, mu):
    """(L, d) hidden -> (L, d/64) uint64 sign bits after centering."""
    bits = (h - mu) > 0
    return np.packbits(bits, axis=-1, bitorder="little").view(np.uint64)


class AngleMemory:
    def __init__(self, layer, mu, tau, beta=1.0, margin=16):
        self.layer, self.mu, self.tau, self.beta, self.margin = layer, mu, tau, beta, margin
        self.code = np.zeros((0, mu.shape[0] // 64), np.uint64)
        self.tok = np.zeros(0, np.int64)
        self.w = np.zeros(0, np.int64)

    def dist(self, c):
        """(L, W) codes -> (L, N) Hamming distances to every stored entry."""
        out = np.empty((len(c), len(self.code)), np.int32)
        for k in range(0, len(c), 256):
            out[k:k + 256] = np.bitwise_count(c[k:k + 256, None, :] ^ self.code[None]).sum(-1)
        return out

    def learn(self, hidden, tokens):
        c = codes(hidden[:-1], self.mu)
        nxt = np.array(tokens[1:], np.int64)
        if len(self.code):
            d = self.dist(c)
            hit = (d <= self.tau) & (self.tok[None, :] != nxt[:, None])
            for j in np.unique(np.nonzero(hit)[1]):
                self.w[j] //= 2
            keep = self.w > 0
            self.code, self.tok, self.w = self.code[keep], self.tok[keep], self.w[keep]
        self.code = np.concatenate([self.code, c])
        self.tok = np.concatenate([self.tok, nxt])
        self.w = np.concatenate([self.w, np.ones(len(nxt), np.int64)])

    def refine(self, hidden, tokens, lp, start=0):
        lp = lp.copy()
        if not len(self.code):
            return lp
        c = codes(hidden[start:len(tokens) - 1], self.mu)
        d = self.dist(c)
        nxt = np.array(tokens[start + 1:], np.int64)
        # attention by angle, sharp: only entries within tau, and within
        # `margin` bits of the closest one, vote
        fire = (d <= self.tau) & (d <= d.min(1, keepdims=True) + self.margin)
        for r in np.nonzero(fire.any(1))[0]:
            w = self.w[fire[r]]
            N = w.sum()
            ct = w[self.tok[fire[r]] == nxt[r]].sum()
            t = start + r
            lp[t] = np.log(max((ct + self.beta * np.exp(lp[t])) / (N + self.beta), 1e-30))
        self.fired = int(fire.any(1).sum())
        return lp
