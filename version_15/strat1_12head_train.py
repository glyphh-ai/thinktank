"""
Strat 1 — 12-head multi-head attention (the real baseline)
==========================================================
Standard transformer attention with 12 heads (head_dim=10), learned
positional embeddings. This is what the ground wire model must beat.

Param count matches 3-phase exactly: the 4,096 params saved by smaller
Wq/Wk/Wv/Wo (128×120 vs 128×128) are spent on pos_emb (32×128).

Run:  python -u strat1_12head_train.py
Out:  strat1_12head_params.npz
"""

import numpy as np
from step1_data import (
    train_data, val_data, vocab_size, BLOCK_SIZE, decode, stoi,
)

C = 128
N_HEADS = 12
HEAD_DIM = 10
PROJ_DIM = N_HEADS * HEAD_DIM  # 120
MLP_HIDDEN = 4 * C
BLOCK = BLOCK_SIZE
EPS = 1e-5


def init_params(rng, vocab=vocab_size):
    def n(*shape, scale):
        return rng.normal(0, scale, size=shape)
    return {
        "tok_emb": n(vocab, C, scale=0.02),
        "pos_emb": n(BLOCK, C, scale=0.02),
        "Wq": n(C, PROJ_DIM, scale=1/np.sqrt(C)),
        "Wk": n(C, PROJ_DIM, scale=1/np.sqrt(C)),
        "Wv": n(C, PROJ_DIM, scale=1/np.sqrt(C)),
        "Wo": n(PROJ_DIM, C, scale=1/np.sqrt(PROJ_DIM)),
        "g1": np.ones(C), "b1": np.zeros(C),
        "g2": np.ones(C), "b2": np.zeros(C),
        "gf": np.ones(C), "bf": np.zeros(C),
        "W1": n(C, MLP_HIDDEN, scale=1/np.sqrt(C)), "b1m": np.zeros(MLP_HIDDEN),
        "W2": n(MLP_HIDDEN, C, scale=1/np.sqrt(MLP_HIDDEN)), "b2m": np.zeros(C),
        "Wh": n(C, vocab, scale=1/np.sqrt(C)), "bh": np.zeros(vocab),
    }


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


def multihead_attention_fwd(x, p):
    B, T, _ = x.shape
    hd = HEAD_DIM
    mask = np.tril(np.ones((T, T), dtype=bool))

    head_outs = []
    head_caches = []
    for h in range(N_HEADS):
        sl = slice(h * hd, (h + 1) * hd)
        wq_h, wk_h, wv_h = p["Wq"][:, sl], p["Wk"][:, sl], p["Wv"][:, sl]
        q = x @ wq_h
        k = x @ wk_h
        v = x @ wv_h
        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(hd)
        scores = np.where(mask, scores, -1e9)
        scores -= scores.max(-1, keepdims=True)
        e = np.exp(scores)
        att = e / e.sum(-1, keepdims=True)
        out = np.matmul(att, v)
        head_outs.append(out)
        head_caches.append((x, q, k, v, att, out, mask, sl))

    concat = np.concatenate(head_outs, axis=-1)
    y = concat @ p["Wo"]
    return y, (head_caches, concat)


def multihead_attention_bwd(dy, cache, p):
    head_caches, concat = cache
    hd = HEAD_DIM

    dWo = np.einsum('btc,btd->cd', concat, dy)
    dconcat = dy @ p["Wo"].T

    dWq = np.zeros_like(p["Wq"])
    dWk = np.zeros_like(p["Wk"])
    dWv = np.zeros_like(p["Wv"])
    dx = np.zeros_like(head_caches[0][0])

    for h in range(N_HEADS):
        x, q, k, v, att, out, mask, sl = head_caches[h]
        wq_h, wk_h, wv_h = p["Wq"][:, sl], p["Wk"][:, sl], p["Wv"][:, sl]
        dout_h = dconcat[:, :, sl]

        dv = np.matmul(att.transpose(0, 2, 1), dout_h)
        datt = np.matmul(dout_h, v.transpose(0, 2, 1))
        dscores = att * (datt - (datt * att).sum(-1, keepdims=True))
        dscores = np.where(mask, dscores, 0.0)
        dscores /= np.sqrt(hd)

        dq = np.matmul(dscores, k)
        dk = np.matmul(dscores.transpose(0, 2, 1), q)

        dWq[:, sl] += np.einsum('btc,btd->cd', x, dq)
        dWk[:, sl] += np.einsum('btc,btd->cd', x, dk)
        dWv[:, sl] += np.einsum('btc,btd->cd', x, dv)
        dx += dq @ wq_h.T + dk @ wk_h.T + dv @ wv_h.T

    return dx, {"Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo}


def forward(p, x_ids, y_ids=None):
    B, T = x_ids.shape
    tok = p["tok_emb"][x_ids]
    pos = p["pos_emb"][:T]
    h0 = tok + pos

    ln1, c_ln1 = layernorm_fwd(h0, p["g1"], p["b1"])
    attn, c_att = multihead_attention_fwd(ln1, p)
    h1 = h0 + attn

    ln2, c_ln2 = layernorm_fwd(h1, p["g2"], p["b2"])
    hid, c_lin1 = linear_fwd(ln2, p["W1"], p["b1m"])
    relu_mask = hid > 0
    act = hid * relu_mask
    mlp, c_lin2 = linear_fwd(act, p["W2"], p["b2m"])
    h2 = h1 + mlp

    lnf, c_lnf = layernorm_fwd(h2, p["gf"], p["bf"])
    logits, c_head = linear_fwd(lnf, p["Wh"], p["bh"])

    cache = dict(x_ids=x_ids, B=B, T=T,
                 c_ln1=c_ln1, c_att=c_att, c_ln2=c_ln2,
                 c_lin1=c_lin1, relu_mask=relu_mask, c_lin2=c_lin2,
                 c_lnf=c_lnf, c_head=c_head)

    if y_ids is None:
        return logits, cache

    flat = logits.reshape(-1, vocab_size)
    flat = flat - flat.max(1, keepdims=True)
    probs = np.exp(flat)
    probs /= probs.sum(1, keepdims=True)
    yf = y_ids.reshape(-1)
    N = yf.shape[0]
    loss = -np.log(probs[np.arange(N), yf] + 1e-12).mean()
    cache["probs"] = probs
    cache["yf"] = yf
    return loss, logits, cache


def backward(p, cache):
    probs, yf = cache["probs"], cache["yf"]
    N = yf.shape[0]
    B, T = cache["B"], cache["T"]
    grads = {k: np.zeros_like(v) for k, v in p.items()}

    dlogits = probs.copy()
    dlogits[np.arange(N), yf] -= 1.0
    dlogits /= N
    dlogits = dlogits.reshape(B, T, vocab_size)

    dlnf, grads["Wh"], grads["bh"] = linear_bwd(dlogits, cache["c_head"])
    dh2, grads["gf"], grads["bf"] = layernorm_bwd(dlnf, cache["c_lnf"])

    dh1 = dh2.copy()
    dact, grads["W2"], grads["b2m"] = linear_bwd(dh2, cache["c_lin2"])
    dhid = dact * cache["relu_mask"]
    dln2, grads["W1"], grads["b1m"] = linear_bwd(dhid, cache["c_lin1"])
    dh1b, grads["g2"], grads["b2"] = layernorm_bwd(dln2, cache["c_ln2"])
    dh1 += dh1b

    dh0 = dh1.copy()
    dln1, gatt = multihead_attention_bwd(dh1, cache["c_att"], p)
    for kk, vv in gatt.items():
        grads[kk] += vv
    dh0b, grads["g1"], grads["b1"] = layernorm_bwd(dln1, cache["c_ln1"])
    dh0 += dh0b

    np.add.at(grads["tok_emb"], cache["x_ids"], dh0)
    grads["pos_emb"][:T] += dh0.sum(0)
    return grads


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


def get_batch(split, rng, B=32, T=BLOCK):
    src = train_data if split == "train" else val_data
    ix = rng.integers(0, len(src) - T, size=B)
    x = np.stack([src[i:i + T] for i in ix])
    y = np.stack([src[i + 1:i + 1 + T] for i in ix])
    return x, y

def evaluate(p, split, rng, B=64):
    x, y = get_batch(split, rng, B=B)
    loss, logits, _ = forward(p, x, y)
    return loss, (logits.argmax(-1) == y).mean()


def sample(p, n=200, seed=0, temperature=0.8):
    rng = np.random.default_rng(seed)
    out = [stoi["<eot>"]]
    for _ in range(n):
        x = np.array([out[-BLOCK:]])
        logits, _ = forward(p, x)
        z = logits[0, -1] / temperature
        probs = np.exp(z - z.max())
        probs /= probs.sum()
        nxt = int(rng.choice(len(probs), p=probs))
        out.append(nxt)
    return decode(out[1:])


def grad_check(seed=1, n_per_param=6, h=1e-5):
    rng = np.random.default_rng(seed)
    p = init_params(rng)
    x = np.array([[3, 8, 1, 8, 2], [5, 16, 7, 1, 6]])
    y = np.array([[8, 1, 8, 2, 1], [16, 7, 1, 6, 25]])
    _, _, cache = forward(p, x, y)
    grads = backward(p, cache)
    worst = 0.0
    for k in p:
        flat = p[k].reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_per_param, flat.size), replace=False)
        for i in idxs:
            old = flat[i]
            flat[i] = old + h; lp, _, _ = forward(p, x, y)
            flat[i] = old - h; lm, _, _ = forward(p, x, y)
            flat[i] = old
            num = (lp - lm) / (2 * h)
            ana = grads[k].reshape(-1)[i]
            worst = max(worst, abs(num - ana) / max(1e-9, abs(num) + abs(ana)))
    return worst


STEPS = 8000
B = 32
LR = 2e-3

if __name__ == "__main__":
    print("== gradient check ==")
    err = grad_check()
    print(f"max relative error: {err:.2e} ({'OK' if err < 1e-4 else 'FAIL'})\n")

    p_count = sum(v.size for v in init_params(np.random.default_rng(0)).values())
    print(f"vocab={vocab_size}  C={C}  heads={N_HEADS}  head_dim={HEAD_DIM}  "
          f"T={BLOCK}  params={p_count:,}\n")

    rng = np.random.default_rng(0)
    p = init_params(rng)
    opt = Adam(p, lr=LR)
    eval_rng = np.random.default_rng(123)

    for step in range(1, STEPS + 1):
        x, y = get_batch("train", rng, B=B)
        loss, _, cache = forward(p, x, y)
        grads = backward(p, cache)
        opt.step(p, grads)
        if step % 500 == 0 or step == 1:
            tr_loss, tr_acc = evaluate(p, "train", eval_rng)
            va_loss, va_acc = evaluate(p, "val", eval_rng)
            print(f"step {step:4d} | train loss {tr_loss:6.3f} acc {tr_acc:5.1%}"
                  f" | val loss {va_loss:6.3f} acc {va_acc:5.1%}")

    np.savez("strat1_12head_params.npz", **p)
    print("\nsaved -> strat1_12head_params.npz")

    for temp in (0.7, 0.9):
        print(f"\n--- sample (temperature={temp}) ---")
        print(sample(p, n=150, seed=42, temperature=temp))
