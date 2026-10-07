"""
Step 3 — THE BASE GPT (multi-layer, multi-head, manual backprop, AdamW)
======================================================================
The version_0_base loop grown into a real small GPT, still NumPy only with
every gradient written out by hand:

    ids (B,T)
      -> token + position embeddings                       (B,T,C)
      -> N_LAYER x [ LN -> multi-head causal attention -> + residual ]
                   [ LN -> MLP (GELU)                  -> + residual ]
      -> final LayerNorm
      -> head = tok_emb^T  (weight-tied)                   (B,T,vocab)
      -> cross-entropy vs the shifted targets

What changed vs version_0_base (1 block, 1 head, ReLU, Adam, fixed lr):
  - 4 blocks x 4 heads, C=256, T=64 context
  - GELU, weight-tied head, GPT-2-style init (residual outputs / sqrt(2L))
  - AdamW (decoupled weight decay on matrices), grad-norm clip 1.0,
    linear warmup + cosine decay
  - float32 throughout (2x faster than float64 on CPU)

Run:  python step3_gpt.py          # short smoke train + sample
      python test_train.py         # gradient check + it-learns check
      python strat0_train.py       # the 5k-step base run
"""

import numpy as np
from step1_data import vocab_size, BLOCK_SIZE, decode, EOT, get_batch

# ---- Hyperparameters ---------------------------------------------------------
N_LAYER = 4
N_HEAD = 4
C = 256              # embedding dim
HEAD_DIM = C // N_HEAD
MLP_HIDDEN = 4 * C
BLOCK = BLOCK_SIZE   # T (context length)
EPS = 1e-5
DT = np.float32


# ============================================================================
# Parameters
# ============================================================================
def init_params(rng, vocab=vocab_size, C=C, n_layer=N_LAYER):
    def n(*shape, scale=0.02):
        return rng.normal(0, scale, size=shape).astype(DT)
    resid = 0.02 / np.sqrt(2 * n_layer)    # scale for projections into the residual
    p = {"tok_emb": n(vocab, C), "pos_emb": n(BLOCK, C, scale=0.01),
         "gf": np.ones(C, DT), "bf": np.zeros(C, DT)}
    for l in range(n_layer):
        p.update({
            f"{l}.g1": np.ones(C, DT), f"{l}.b1": np.zeros(C, DT),
            f"{l}.Wqkv": n(C, 3 * C), f"{l}.Wo": n(C, C, scale=resid),
            f"{l}.g2": np.ones(C, DT), f"{l}.b2": np.zeros(C, DT),
            f"{l}.W1": n(C, MLP_HIDDEN), f"{l}.b1m": np.zeros(MLP_HIDDEN, DT),
            f"{l}.W2": n(MLP_HIDDEN, C, scale=resid), f"{l}.b2m": np.zeros(C, DT),
        })
    return p

def n_params(p):
    return sum(v.size for v in p.values())


# ============================================================================
# Building blocks (each forward returns value + cache for backward)
# ============================================================================
def layernorm_fwd(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    istd = 1.0 / np.sqrt(var + EPS)
    xhat = (x - mu) * istd
    return g * xhat + b, (xhat, istd, g)

def layernorm_bwd(dout, cache):
    xhat, istd, g = cache
    Cdim = xhat.shape[-1]
    dg = (dout * xhat).reshape(-1, Cdim).sum(0)
    db = dout.reshape(-1, Cdim).sum(0)
    dxhat = dout * g
    dx = istd * (dxhat
                 - dxhat.mean(-1, keepdims=True)
                 - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db

def matmul_bwd(dout, x, W):
    """y = x @ W  ->  (dx, dW)."""
    dW = x.reshape(-1, W.shape[0]).T @ dout.reshape(-1, W.shape[1])
    return dout @ W.T, dW

_GA = np.float32(np.sqrt(2 / np.pi))

def gelu_fwd(x):
    t = np.tanh(_GA * (x + 0.044715 * x * x * x))   # x*x*x: ~50x faster than x**3
    return 0.5 * x * (1 + t), (x, t)

def gelu_bwd(dout, cache):
    x, t = cache
    dt = (1 - t * t) * _GA * (1 + 3 * 0.044715 * x * x)
    return dout * (0.5 * (1 + t) + 0.5 * x * dt)


# ============================================================================
# Multi-head causal self-attention
# ============================================================================
_MASKS = {}

def _mask(T):
    if T not in _MASKS:
        _MASKS[T] = np.triu(np.full((T, T), -1e9, DT), k=1)   # 0 on/below diag
    return _MASKS[T]

def attention_fwd(x, Wqkv, Wo):
    B, T, _ = x.shape
    qkv = x @ Wqkv                                           # (B,T,3C)
    qkv = qkv.reshape(B, T, 3, N_HEAD, HEAD_DIM).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]                         # (B,H,T,D)
    scale = DT(1 / np.sqrt(HEAD_DIM))
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale + _mask(T)  # (B,H,T,T)
    scores -= scores.max(-1, keepdims=True)
    att = np.exp(scores)
    att /= att.sum(-1, keepdims=True)
    out = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)   # (B,T,C)
    return out @ Wo, (x, q, k, v, att, out)

def attention_bwd(dy, cache, Wqkv, Wo):
    x, q, k, v, att, out = cache
    B, T, _ = x.shape
    dout, dWo = matmul_bwd(dy, out, Wo)
    dout = dout.reshape(B, T, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)  # (B,H,T,D)
    dv = att.transpose(0, 1, 3, 2) @ dout
    datt = dout @ v.transpose(0, 1, 3, 2)
    # softmax backward (masked entries have att=0 so their grad is 0 already)
    dscores = att * (datt - (datt * att).sum(-1, keepdims=True))
    dscores *= DT(1 / np.sqrt(HEAD_DIM))
    dq = dscores @ k
    dk = dscores.transpose(0, 1, 3, 2) @ q
    dqkv = np.stack([dq, dk, dv]).transpose(1, 3, 0, 2, 4).reshape(B, T, 3 * C)
    dx, dWqkv = matmul_bwd(dqkv, x, Wqkv)
    return dx, dWqkv, dWo


# ============================================================================
# Full forward / backward
# ============================================================================
def forward(p, x_ids, y_ids=None, n_layer=N_LAYER):
    B, T = x_ids.shape
    h = p["tok_emb"][x_ids] + p["pos_emb"][:T]               # (B,T,C)
    caches = []
    for l in range(n_layer):
        ln1, c_ln1 = layernorm_fwd(h, p[f"{l}.g1"], p[f"{l}.b1"])
        attn, c_att = attention_fwd(ln1, p[f"{l}.Wqkv"], p[f"{l}.Wo"])
        h = h + attn
        ln2, c_ln2 = layernorm_fwd(h, p[f"{l}.g2"], p[f"{l}.b2"])
        hid = ln2 @ p[f"{l}.W1"] + p[f"{l}.b1m"]
        act, c_gelu = gelu_fwd(hid)
        h = h + act @ p[f"{l}.W2"] + p[f"{l}.b2m"]
        caches.append((c_ln1, c_att, c_ln2, ln2, c_gelu, act))
    lnf, c_lnf = layernorm_fwd(h, p["gf"], p["bf"])
    logits = lnf @ p["tok_emb"].T                            # (B,T,V)
    cache = dict(x_ids=x_ids, caches=caches, c_lnf=c_lnf, lnf=lnf)
    if y_ids is None:
        return logits, cache

    flat = logits.reshape(-1, logits.shape[-1])
    flat = flat - flat.max(1, keepdims=True)
    probs = np.exp(flat)
    probs /= probs.sum(1, keepdims=True)
    yf = y_ids.reshape(-1)
    loss = float(-np.log(probs[np.arange(len(yf)), yf] + 1e-12).mean())
    cache["probs"], cache["yf"] = probs, yf
    return loss, logits, cache


def backward(p, cache, n_layer=N_LAYER):
    probs, yf, x_ids = cache["probs"], cache["yf"], cache["x_ids"]
    B, T = x_ids.shape
    g = {}

    dlogits = probs                                          # reuse buffer
    dlogits[np.arange(len(yf)), yf] -= 1.0
    dlogits /= len(yf)
    lnf = cache["lnf"].reshape(-1, C)
    g["tok_emb"] = dlogits.T @ lnf                           # tied head
    dlnf = (dlogits @ p["tok_emb"]).reshape(B, T, C)
    dh, g["gf"], g["bf"] = layernorm_bwd(dlnf, cache["c_lnf"])

    for l in reversed(range(n_layer)):
        c_ln1, c_att, c_ln2, ln2, c_gelu, act = cache["caches"][l]
        # MLP residual
        g[f"{l}.b2m"] = dh.reshape(-1, C).sum(0)
        dact, g[f"{l}.W2"] = matmul_bwd(dh, act, p[f"{l}.W2"])
        dhid = gelu_bwd(dact, c_gelu)
        g[f"{l}.b1m"] = dhid.reshape(-1, MLP_HIDDEN).sum(0)
        dln2, g[f"{l}.W1"] = matmul_bwd(dhid, ln2, p[f"{l}.W1"])
        dx, g[f"{l}.g2"], g[f"{l}.b2"] = layernorm_bwd(dln2, c_ln2)
        dh = dh + dx
        # attention residual
        dln1, g[f"{l}.Wqkv"], g[f"{l}.Wo"] = attention_bwd(
            dh, c_att, p[f"{l}.Wqkv"], p[f"{l}.Wo"])
        dx, g[f"{l}.g1"], g[f"{l}.b1"] = layernorm_bwd(dln1, c_ln1)
        dh = dh + dx

    np.add.at(g["tok_emb"], x_ids, dh)
    g["pos_emb"] = np.zeros_like(p["pos_emb"])
    g["pos_emb"][:T] = dh.sum(0)
    return g


# ============================================================================
# AdamW + schedule
# ============================================================================
class AdamW:
    """Adam with decoupled weight decay on matrices (ndim >= 2) only."""
    def __init__(self, params, betas=(0.9, 0.95), eps=1e-8, wd=0.1):
        self.b1, self.b2, self.eps, self.wd = betas[0], betas[1], eps, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for k, w in params.items():
            gk = grads[k]
            self.m[k] *= self.b1; self.m[k] += (1 - self.b1) * gk
            self.v[k] *= self.b2; self.v[k] += (1 - self.b2) * gk * gk
            if w.ndim >= 2 and self.wd:
                w *= DT(1 - lr * self.wd)
            w -= DT(lr) * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + DT(self.eps))

def clip_grads(grads, max_norm=1.0):
    norm = float(np.sqrt(sum(float((gk * gk).sum()) for gk in grads.values())))
    if norm > max_norm:
        s = DT(max_norm / (norm + 1e-6))
        for gk in grads.values():
            gk *= s
    return norm

def lr_at(step, steps, max_lr, warmup=200, min_ratio=0.1):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    frac = (step - warmup) / max(1, steps - warmup)
    return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * frac)))


# ============================================================================
# Metrics, training, sampling
# ============================================================================
def evaluate(p, split, rng, batches=4, B=64):
    """Mean (loss, accuracy) over a few freshly drawn batches."""
    ls, accs = [], []
    for _ in range(batches):
        x, y = get_batch(split, rng, B=B, T=BLOCK)
        loss, logits, _ = forward(p, x, y)
        ls.append(loss)
        accs.append(float((logits.argmax(-1) == y).mean()))
    return float(np.mean(ls)), float(np.mean(accs))


def train(steps=5000, B=32, lr=2e-3, seed=0, log_every=250, verbose=True,
          p=None, on_log=None):
    rng = np.random.default_rng(seed)
    p = init_params(rng) if p is None else p
    opt = AdamW(p)
    history = []
    for step in range(steps):
        x, y = get_batch("train", rng, B=B, T=BLOCK)
        loss, _, cache = forward(p, x, y)
        grads = backward(p, cache)
        gnorm = clip_grads(grads)
        cur_lr = lr_at(step, steps, lr)
        opt.step(p, grads, cur_lr)
        if verbose and ((step + 1) % log_every == 0 or step == 0):
            eval_rng = np.random.default_rng(123)       # same eval batches each time
            tr = evaluate(p, "train", eval_rng)
            va = evaluate(p, "val", eval_rng)
            history.append((step + 1, *tr, *va))
            print(f"step {step+1:5d} | lr {cur_lr:.2e} | gnorm {gnorm:5.2f} | "
                  f"train loss {tr[0]:.3f} acc {tr[1]:5.1%} | "
                  f"val loss {va[0]:.3f} acc {va[1]:5.1%}", flush=True)
            if on_log:
                on_log(step + 1, p)
    return p, history


def sample(p, n=200, seed=0, temperature=0.8, top_k=40, prompt_ids=None):
    """Autoregressive generation (temperature + top-k).

    Starts at a story boundary (<eot>) unless prompt_ids is given.
    """
    rng = np.random.default_rng(seed)
    out = list(prompt_ids) if prompt_ids else [EOT]
    start = len(out)
    for _ in range(n):
        logits, _ = forward(p, np.array([out[-BLOCK:]]))
        z = logits[0, -1].astype(np.float64) / temperature
        if top_k:
            z[z < np.partition(z, -top_k)[-top_k]] = -np.inf
        probs = np.exp(z - z.max())
        probs /= probs.sum()
        out.append(int(rng.choice(len(probs), p=probs)))
    return decode(out if prompt_ids else out[start:])


if __name__ == "__main__":
    print(f"vocab={vocab_size}  L={N_LAYER}  H={N_HEAD}  C={C}  T={BLOCK}  "
          f"params={n_params(init_params(np.random.default_rng(0))):,}\n")
    p, _ = train(steps=300, log_every=50)
    print("\n--- sample after 300 steps ---")
    print(sample(p, n=80))
