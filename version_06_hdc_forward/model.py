"""
model.py — self-contained word-level NumPy GPT (the sacred baseline + proposal net).
====================================================================================
Vendored from thinktank/version_05_lattice/step3_gpt.py (from-scratch GPT: token+
position embeddings -> 1 transformer block -> LM head -> cross-entropy -> manual
backprop -> Adam). Refactored to be self-contained: C (embed dim) and V (vocab)
are read from the parameter shapes instead of module globals, so one process can
hold several models / vocabs.

This net plays two roles downstream:
  1. the BASELINE  (baseline.py)      — vanilla LM loss / perplexity, the sacred delta.
  2. the PROPOSER  (forward_hdc.py)    — supplies next-word logits that the HDC
                                         similarity gate re-scores at word boundaries.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-5


# ---- building blocks (identical math to v05 step3_gpt) ----------------------
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
    dx = istd * (dxhat - dxhat.mean(-1, keepdims=True)
                 - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


def linear_fwd(x, W, b):
    return x @ W + b, (x, W)


def linear_bwd(dout, cache):
    x, W = cache
    Cin = W.shape[0]
    dW = x.reshape(-1, Cin).T @ dout.reshape(-1, W.shape[1])
    db = dout.reshape(-1, W.shape[1]).sum(0)
    dx = dout @ W.T
    return dx, dW, db


def attention_fwd(x, p):
    B, T, C = x.shape
    q = x @ p["Wq"]; k = x @ p["Wk"]; v = x @ p["Wv"]
    scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(C)
    mask = np.tril(np.ones((T, T), dtype=bool))
    scores = np.where(mask, scores, -1e9)
    scores -= scores.max(-1, keepdims=True)
    e = np.exp(scores)
    att = e / e.sum(-1, keepdims=True)
    out = np.matmul(att, v)
    y = out @ p["Wo"]
    return y, (x, q, k, v, att, out, mask)


def attention_bwd(dy, cache, p):
    x, q, k, v, att, out, mask = cache
    C = x.shape[-1]
    dWo = np.einsum('btc,btd->cd', out, dy)
    dout = dy @ p["Wo"].T
    dv = np.matmul(att.transpose(0, 2, 1), dout)
    datt = np.matmul(dout, v.transpose(0, 2, 1))
    dscores = att * (datt - (datt * att).sum(-1, keepdims=True))
    dscores = np.where(mask, dscores, 0.0)
    dscores /= np.sqrt(C)
    dq = np.matmul(dscores, k)
    dk = np.matmul(dscores.transpose(0, 2, 1), q)
    dWq = np.einsum('btc,btd->cd', x, dq)
    dWk = np.einsum('btc,btd->cd', x, dk)
    dWv = np.einsum('btc,btd->cd', x, dv)
    dx = dq @ p["Wq"].T + dk @ p["Wk"].T + dv @ p["Wv"].T
    return dx, {"Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo}


# ---- params -----------------------------------------------------------------
def init_params(rng, vocab, T, C=64):
    def n(*shape, sc):
        return rng.normal(0, sc, size=shape)
    H = 4 * C
    return {
        "tok_emb": n(vocab, C, sc=0.02), "pos_emb": n(T, C, sc=0.02),
        "Wq": n(C, C, sc=1/np.sqrt(C)), "Wk": n(C, C, sc=1/np.sqrt(C)),
        "Wv": n(C, C, sc=1/np.sqrt(C)), "Wo": n(C, C, sc=1/np.sqrt(C)),
        "g1": np.ones(C), "b1": np.zeros(C),
        "g2": np.ones(C), "b2": np.zeros(C),
        "gf": np.ones(C), "bf": np.zeros(C),
        "W1": n(C, H, sc=1/np.sqrt(C)), "b1m": np.zeros(H),
        "W2": n(H, C, sc=1/np.sqrt(H)), "b2m": np.zeros(C),
        "Wh": n(C, vocab, sc=1/np.sqrt(C)), "bh": np.zeros(vocab),
    }


def forward(p, x_ids, y_ids=None):
    B, T = x_ids.shape
    V = p["Wh"].shape[1]
    h0 = p["tok_emb"][x_ids] + p["pos_emb"][:T]
    ln1, c_ln1 = layernorm_fwd(h0, p["g1"], p["b1"])
    attn, c_att = attention_fwd(ln1, p)
    h1 = h0 + attn
    ln2, c_ln2 = layernorm_fwd(h1, p["g2"], p["b2"])
    hid, c_lin1 = linear_fwd(ln2, p["W1"], p["b1m"])
    relu_mask = hid > 0
    act = hid * relu_mask
    mlp, c_lin2 = linear_fwd(act, p["W2"], p["b2m"])
    h2 = h1 + mlp
    lnf, c_lnf = layernorm_fwd(h2, p["gf"], p["bf"])
    logits, c_head = linear_fwd(lnf, p["Wh"], p["bh"])
    cache = dict(x_ids=x_ids, B=B, T=T, V=V, c_ln1=c_ln1, c_att=c_att, c_ln2=c_ln2,
                 c_lin1=c_lin1, relu_mask=relu_mask, c_lin2=c_lin2, c_lnf=c_lnf, c_head=c_head)
    if y_ids is None:
        return logits, cache
    flat = logits.reshape(-1, V)
    flat = flat - flat.max(1, keepdims=True)
    probs = np.exp(flat); probs /= probs.sum(1, keepdims=True)
    yf = y_ids.reshape(-1)
    N = yf.shape[0]
    loss = -np.log(probs[np.arange(N), yf] + 1e-12).mean()
    cache["probs"] = probs; cache["yf"] = yf
    return loss, logits, cache


def backward(p, cache):
    probs, yf = cache["probs"], cache["yf"]
    N = yf.shape[0]; B, T, V = cache["B"], cache["T"], cache["V"]
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    dlogits = probs.copy()
    dlogits[np.arange(N), yf] -= 1.0
    dlogits /= N
    dlogits = dlogits.reshape(B, T, V)
    dlnf, grads["Wh"], grads["bh"] = linear_bwd(dlogits, cache["c_head"])
    dh2, grads["gf"], grads["bf"] = layernorm_bwd(dlnf, cache["c_lnf"])
    dh1 = dh2.copy()
    dact, grads["W2"], grads["b2m"] = linear_bwd(dh2, cache["c_lin2"])
    dhid = dact * cache["relu_mask"]
    dln2, grads["W1"], grads["b1m"] = linear_bwd(dhid, cache["c_lin1"])
    dh1b, grads["g2"], grads["b2"] = layernorm_bwd(dln2, cache["c_ln2"])
    dh1 += dh1b
    dh0 = dh1.copy()
    dln1, gatt = attention_bwd(dh1, cache["c_att"], p)
    for kk, vv in gatt.items():
        grads[kk] += vv
    dh0b, grads["g1"], grads["b1"] = layernorm_bwd(dln1, cache["c_ln1"])
    dh0 += dh0b
    np.add.at(grads["tok_emb"], cache["x_ids"], dh0)
    grads["pos_emb"][:T] += dh0.sum(0)
    return grads


def _trunk_backward(dlnf, cache, p):
    """Backprop from the final-LN output down to the embeddings (shared by both heads)."""
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    dh2, grads["gf"], grads["bf"] = layernorm_bwd(dlnf, cache["c_lnf"])
    dh1 = dh2.copy()
    dact, grads["W2"], grads["b2m"] = linear_bwd(dh2, cache["c_lin2"])
    dhid = dact * cache["relu_mask"]
    dln2, grads["W1"], grads["b1m"] = linear_bwd(dhid, cache["c_lin1"])
    dh1b, grads["g2"], grads["b2"] = layernorm_bwd(dln2, cache["c_ln2"])
    dh1 += dh1b
    dh0 = dh1.copy()
    dln1, gatt = attention_bwd(dh1, cache["c_att"], p)
    for kk, vv in gatt.items():
        grads[kk] += vv
    dh0b, grads["g1"], grads["b1"] = layernorm_bwd(dln1, cache["c_ln1"])
    dh0 += dh0b
    np.add.at(grads["tok_emb"], cache["x_ids"], dh0)
    grads["pos_emb"][:cache["T"]] += dh0.sum(0)
    return grads


# ---- dual head (LM + Ada role), the v05 in-loop objective -------------------
def add_role_head(p, rng, n_role):
    C = p["tok_emb"].shape[1]
    p["Wr"] = rng.normal(0, 1 / np.sqrt(C), size=(C, n_role))
    p["br"] = np.zeros(n_role)
    return p


def _ce(logits, y):
    V = logits.shape[-1]
    flat = logits.reshape(-1, V)
    flat = flat - flat.max(1, keepdims=True)
    probs = np.exp(flat); probs /= probs.sum(1, keepdims=True)
    yf = y.reshape(-1)
    loss = -np.log(probs[np.arange(yf.shape[0]), yf] + 1e-12).mean()
    return loss, probs, yf


def forward_dual(p, x_ids, yw=None, yr=None):
    """Trunk + LM head + role head. Role logits read the same final hidden as LM."""
    lm_logits, cache = forward(p, x_ids)
    lnf = cache["c_head"][0]
    role_logits = lnf @ p["Wr"] + p["br"]
    cache["lnf"] = lnf
    if yw is None:
        return lm_logits, role_logits, cache
    lm_loss, lm_probs, ywf = _ce(lm_logits, yw)
    r_loss, r_probs, yrf = _ce(role_logits, yr)
    cache.update(lm_probs=lm_probs, ywf=ywf, r_probs=r_probs, yrf=yrf)
    return lm_loss, r_loss, lm_logits, role_logits, cache


def backward_dual(p, cache, lam=1.0):
    B, T, V = cache["B"], cache["T"], cache["V"]
    NR = p["br"].shape[0]
    # LM head
    dlm = cache["lm_probs"].copy(); dlm[np.arange(cache["ywf"].shape[0]), cache["ywf"]] -= 1.0
    dlm /= cache["ywf"].shape[0]
    dlnf_lm, dWh, dbh = linear_bwd(dlm.reshape(B, T, V), cache["c_head"])
    # role head
    dr = cache["r_probs"].copy(); dr[np.arange(cache["yrf"].shape[0]), cache["yrf"]] -= 1.0
    dr /= cache["yrf"].shape[0]; dr *= lam
    dlnf_r, dWr, dbr = linear_bwd(dr.reshape(B, T, NR), (cache["lnf"], p["Wr"]))
    # shared trunk
    grads = _trunk_backward(dlnf_lm + dlnf_r, cache, p)
    grads["Wh"], grads["bh"] = dWh, dbh
    grads["Wr"], grads["br"] = dWr, dbr
    return grads


def train_dual(ids, rids, vocab_size, n_role, T=16, C=64, steps=1500, B=64,
               lr=3e-3, lam=1.0, seed=0, log_every=250):
    rng = np.random.default_rng(seed)
    p = init_params(rng, vocab_size, T, C)
    add_role_head(p, rng, n_role)
    opt = Adam(p, lr=lr)
    for step in range(1, steps + 1):
        ix = rng.integers(0, len(ids) - T - 1, size=B)
        x = np.stack([ids[i:i + T] for i in ix])
        yw = np.stack([ids[i + 1:i + 1 + T] for i in ix])
        yr = np.stack([rids[i + 1:i + 1 + T] for i in ix])
        lm_loss, r_loss, _, _, cache = forward_dual(p, x, yw, yr)
        opt.step(p, backward_dual(p, cache, lam=lam))
        if log_every and (step % log_every == 0 or step == 1):
            print(f"  step {step:5d}  lm {lm_loss:.3f} (ppl {np.exp(lm_loss):6.2f})  role {r_loss:.3f}")
    return p


def next_dual(p, context_ids, T):
    """Next-word logits AND next-word role logits given a 1-D context."""
    ctx = np.asarray(context_ids[-T:], dtype=np.int64)[None, :]
    lm_logits, role_logits, _ = forward_dual(p, ctx)
    return lm_logits[0, -1], role_logits[0, -1]


class Adam:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.99), eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ---- training + generation helpers ------------------------------------------
def sample_batch(ids, T, B, rng):
    ix = rng.integers(0, len(ids) - T - 1, size=B)
    x = np.stack([ids[i:i + T] for i in ix])
    y = np.stack([ids[i + 1:i + 1 + T] for i in ix])
    return x, y


def train(ids, vocab_size, T=16, C=64, steps=1500, B=64, lr=3e-3, seed=0, log_every=250):
    rng = np.random.default_rng(seed)
    p = init_params(rng, vocab_size, T, C)
    opt = Adam(p, lr=lr)
    for step in range(1, steps + 1):
        x, y = sample_batch(ids, T, B, rng)
        loss, _, cache = forward(p, x, y)
        opt.step(p, backward(p, cache))
        if log_every and (step % log_every == 0 or step == 1):
            print(f"  step {step:5d}  loss {loss:.4f}  ppl {np.exp(loss):7.2f}")
    return p


def next_logits(p, context_ids, T):
    """Next-word logits given a 1-D context (last T tokens used)."""
    ctx = np.asarray(context_ids[-T:], dtype=np.int64)[None, :]
    logits, _ = forward(p, ctx)
    return logits[0, -1]


def eval_ppl(p, ids, T, rng, batches=20, B=128):
    losses = []
    for _ in range(batches):
        x, y = sample_batch(ids, T, B, rng)
        loss, _, _ = forward(p, x, y)
        losses.append(loss)
    return float(np.mean(losses))
