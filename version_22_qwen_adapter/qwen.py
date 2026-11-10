"""qwen.py — Qwen3 through MLX, opened up: log-probs plus hidden states.

One forward pass returns, for every position, log p(next token) under Qwen
and the hidden state after chosen layers (what the angle memory keys on).
"""

import glob
import os

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache

NAME = os.environ.get("QWEN", "Qwen3-1.7B-4bit")     # QWEN=Qwen3-14B-4bit for a bigger one


def path(name=NAME):
    return glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--mlx-community--{name}/snapshots/*"))[0]


class Qwen:
    def __init__(self, name=NAME):
        self.model, self.tok = load(path(name))
        self.inner = self.model.model
        self.n_layers = len(self.inner.layers)

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def forward(self, x, layers=(), cache=None):
        """x: (B, L) int tokens -> logits (B, L, V), {layer: hidden (B, L, d)}."""
        h = self.inner.embed_tokens(x)
        if cache is None:
            mask = "causal" if x.shape[1] > 1 else None
            cache = [None] * self.n_layers
        else:
            mask = cache[0].make_mask(x.shape[1], return_array=False, window_size=None)
        hid = {}
        for i, (layer, c) in enumerate(zip(self.inner.layers, cache)):
            h = layer(h, mask, c)
            if i in layers:
                hid[i] = h
        out = self.inner.norm(h)
        if "final" in layers:
            hid["final"] = out
        return self.model.model.embed_tokens.as_linear(out) \
            if self.model.args.tie_word_embeddings else self.model.lm_head(out), hid

    def run(self, seqs, layers=(), batch=32, prefix_cache=None):
        """Right-pad seqs, run, return per-seq (logp of each actual next token
        (L-1,), {layer: hidden (L, d) float32})."""
        out = []
        for k in range(0, len(seqs), batch):
            chunk = seqs[k:k + batch]
            L = max(map(len, chunk))
            x = np.zeros((len(chunk), L), np.int32)
            for r, s in enumerate(chunk):
                x[r, :len(s)] = s
            cache = batched(prefix_cache, len(chunk)) if prefix_cache else None
            logits, hid = self.forward(mx.array(x), layers, cache)
            logits = logits.astype(mx.float32)
            lse = mx.logsumexp(logits[:, :-1], axis=-1)
            tgt = mx.take_along_axis(logits[:, :-1], mx.array(x[:, 1:])[..., None], -1)[..., 0]
            lp = np.array(tgt - lse)
            hs ={key: np.array(v.astype(mx.float32)) for key, v in hid.items()}
            for r, s in enumerate(chunk):
                out.append((lp[r, :len(s) - 1], {key: v[r, :len(s)] for key, v in hs.items()}))
        return out


def prefix_cache(q, tokens):
    cache = [KVCache() for _ in range(q.n_layers)]
    q.forward(mx.array(np.array(tokens, np.int32)[None]), (), cache)
    mx.eval([c.keys for c in cache])
    return cache


def batched(cache, B):
    out = []
    for c in cache:
        n = KVCache()
        k, v = c.keys[..., :c.offset, :], c.values[..., :c.offset, :]
        n.keys, n.values = mx.repeat(k, B, axis=0), mx.repeat(v, B, axis=0)
        n.offset = c.offset
        out.append(n)
    return out
