"""
Strat 2 — hybrid: phase geometry + learned content differentiation
===================================================================
3-phase π-pos attention with lightweight per-phase learned adapters.
Tests whether a small learned perturbation closes the gap to standard
multi-head without full per-head projections.

Three variants:
  A) Per-phase bias on Q         (384 params)
  B) Per-phase LoRA rank-4 on Q  (3,072 params)
  C) Per-phase output scale      (384 params)

For comparison, standard 3-head spends 48,384 params on per-head projections.

Run:  python -u strat2_hybrid_train.py
"""

import numpy as np
from step1_data import (
    train_data, val_data, vocab_size, BLOCK_SIZE, decode, stoi,
)

C = 128
N_PHASES = 3
MLP_HIDDEN = 4 * C
BLOCK = BLOCK_SIZE
EPS = 1e-5
LORA_RANK = 4


# ============================================================================
# π-frequency positional encoding
# ============================================================================
def multi_phase_pi_pos(T, C, n_phases=N_PHASES):
    pos = np.arange(T)[:, None].astype(np.float64)
    dim = np.arange(C)[None, :].astype(np.float64)
    freq = 1.0 / (np.pi ** (2.0 * (dim // 2) / C))
    return np.stack([
        np.where(dim % 2 == 0,
                 np.sin(pos * freq + 2.0 * np.pi * k / n_phases),
                 np.cos(pos * freq + 2.0 * np.pi * k / n_phases))
        for k in range(n_phases)
    ])


# ============================================================================
# Parameters for each variant
# ============================================================================
def init_params(rng, variant, vocab=vocab_size):
    def n(*shape, scale):
        return rng.normal(0, scale, size=shape)
    p = {
        "tok_emb": n(vocab, C, scale=0.02),
        "Wq": n(C, C, scale=1/np.sqrt(C)),
        "Wk": n(C, C, scale=1/np.sqrt(C)),
        "Wv": n(C, C, scale=1/np.sqrt(C)),
        "Wo": n(C, C, scale=1/np.sqrt(C)),
        "g1": np.ones(C), "b1": np.zeros(C),
        "g2": np.ones(C), "b2": np.zeros(C),
        "gf": np.ones(C), "bf": np.zeros(C),
        "W1": n(C, MLP_HIDDEN, scale=1/np.sqrt(C)), "b1m": np.zeros(MLP_HIDDEN),
        "W2": n(MLP_HIDDEN, C, scale=1/np.sqrt(MLP_HIDDEN)), "b2m": np.zeros(C),
        "Wh": n(C, vocab, scale=1/np.sqrt(C)), "bh": np.zeros(vocab),
    }
    if variant == "A":
        for phi in range(N_PHASES):
            p[f"qbias_{phi}"] = np.zeros(C)
    elif variant == "B":
        for phi in range(N_PHASES):
            p[f"lora_down_{phi}"] = n(C, LORA_RANK, scale=1/np.sqrt(C))
            p[f"lora_up_{phi}"] = np.zeros((LORA_RANK, C))
    elif variant == "C":
        for phi in range(N_PHASES):
            p[f"out_scale_{phi}"] = np.ones(C)
    return p


# ============================================================================
# Building blocks
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


# ============================================================================
# 3-phase attention with per-phase adapters
# ============================================================================
def attention_fwd(ln_tok, positions, p, variant):
    B, T, _ = ln_tok.shape
    mask = np.tril(np.ones((T, T), dtype=bool))

    phase_outs = []
    phase_caches = []
    for phi in range(N_PHASES):
        x = ln_tok + positions[phi]
        q = x @ p["Wq"]
        k = x @ p["Wk"]
        v = x @ p["Wv"]

        if variant == "A":
            q = q + p[f"qbias_{phi}"]
        elif variant == "B":
            lora = x @ p[f"lora_down_{phi}"] @ p[f"lora_up_{phi}"]
            q = q + lora

        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(C)
        scores = np.where(mask, scores, -1e9)
        scores -= scores.max(-1, keepdims=True)
        e = np.exp(scores)
        att = e / e.sum(-1, keepdims=True)
        out = np.matmul(att, v)

        if variant == "C":
            out = out * p[f"out_scale_{phi}"]

        phase_outs.append(out)
        phase_caches.append((x, q, k, v, att, out, mask))

    combined = sum(phase_outs) / N_PHASES
    y = combined @ p["Wo"]
    return y, (ln_tok, positions, phase_caches, combined, variant)


def attention_bwd(dy, cache, p):
    ln_tok, positions, phase_caches, combined, variant = cache

    dWo = np.einsum('btc,btd->cd', combined, dy)
    dcombined = dy @ p["Wo"].T
    dphase = dcombined / N_PHASES

    dWq = np.zeros_like(p["Wq"])
    dWk = np.zeros_like(p["Wk"])
    dWv = np.zeros_like(p["Wv"])
    dln_tok = np.zeros_like(ln_tok)
    grads = {"Wo": dWo}

    for phi in range(N_PHASES):
        x, q, k, v, att, out, mask = phase_caches[phi]
        dp = dphase.copy()

        if variant == "C":
            scale = p[f"out_scale_{phi}"]
            grads[f"out_scale_{phi}"] = (dp * (out / scale)).reshape(-1, C).sum(0)
            dp = dp * scale

        # out = att @ v
        dv = np.matmul(att.transpose(0, 2, 1), dp)
        datt = np.matmul(dp, v.transpose(0, 2, 1))

        dscores = att * (datt - (datt * att).sum(-1, keepdims=True))
        dscores = np.where(mask, dscores, 0.0)
        dscores /= np.sqrt(C)

        dq = np.matmul(dscores, k)
        dk = np.matmul(dscores.transpose(0, 2, 1), q)

        if variant == "A":
            grads[f"qbias_{phi}"] = dq.reshape(-1, C).sum(0)
        elif variant == "B":
            lora_down = p[f"lora_down_{phi}"]
            lora_up = p[f"lora_up_{phi}"]
            dlora = dq  # dq flows through q = Wq@x + lora
            d_lora_up = np.einsum('btc,btd->cd', x @ lora_down, dlora)
            # reshape for matmul: (B*T, C) @ (C, R) -> (B*T, R), then (B*T, R)^T @ (B*T, C)
            d_lora_down = x.reshape(-1, C).T @ (dlora.reshape(-1, C) @ lora_up.T)
            grads[f"lora_down_{phi}"] = d_lora_down
            grads[f"lora_up_{phi}"] = d_lora_up
            # extra gradient to x from lora path
            dx_lora = (dlora @ lora_up.T) @ lora_down.T
            dln_tok += dx_lora

        dWq += np.einsum('btc,btd->cd', x, dq)
        dWk += np.einsum('btc,btd->cd', x, dk)
        dWv += np.einsum('btc,btd->cd', x, dv)

        dx = dq @ p["Wq"].T + dk @ p["Wk"].T + dv @ p["Wv"].T
        dln_tok += dx

    grads["Wq"] = dWq
    grads["Wk"] = dWk
    grads["Wv"] = dWv
    return dln_tok, grads


# ============================================================================
# Full forward/backward
# ============================================================================
def forward(p, x_ids, y_ids=None, variant="A"):
    B, T = x_ids.shape
    tok = p["tok_emb"][x_ids]
    positions = multi_phase_pi_pos(T, C)

    ln1, c_ln1 = layernorm_fwd(tok, p["g1"], p["b1"])
    attn, c_att = attention_fwd(ln1, positions, p, variant)
    h1 = tok + attn

    ln2, c_ln2 = layernorm_fwd(h1, p["g2"], p["b2"])
    hid, c_lin1 = linear_fwd(ln2, p["W1"], p["b1m"])
    relu_mask = hid > 0
    act = hid * relu_mask
    mlp, c_lin2 = linear_fwd(act, p["W2"], p["b2m"])
    h2 = h1 + mlp

    lnf, c_lnf = layernorm_fwd(h2, p["gf"], p["bf"])
    logits, c_head = linear_fwd(lnf, p["Wh"], p["bh"])

    cache = dict(x_ids=x_ids, B=B, T=T, variant=variant,
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

    dtok = dh1.copy()
    dln1, gatt = attention_bwd(dh1, cache["c_att"], p)
    for kk, vv in gatt.items():
        grads[kk] += vv
    dtok_ln, grads["g1"], grads["b1"] = layernorm_bwd(dln1, cache["c_ln1"])
    dtok += dtok_ln

    np.add.at(grads["tok_emb"], cache["x_ids"], dtok)
    return grads


# ============================================================================
# Training
# ============================================================================
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

def evaluate(p, split, rng, variant, B=64):
    x, y = get_batch(split, rng, B=B)
    loss, logits, _ = forward(p, x, y, variant=variant)
    return loss, (logits.argmax(-1) == y).mean()


def grad_check(variant, seed=1, h=1e-5):
    rng = np.random.default_rng(seed)
    p = init_params(rng, variant)
    x = np.array([[3, 8, 1, 8, 2], [5, 16, 7, 1, 6]])
    y = np.array([[8, 1, 8, 2, 1], [16, 7, 1, 6, 25]])
    _, _, cache = forward(p, x, y, variant=variant)
    grads = backward(p, cache)
    worst = 0.0
    for k in p:
        flat = p[k].reshape(-1)
        idxs = rng.choice(flat.size, size=min(6, flat.size), replace=False)
        for i in idxs:
            old = flat[i]
            flat[i] = old + h; lp, _, _ = forward(p, x, y, variant=variant)
            flat[i] = old - h; lm, _, _ = forward(p, x, y, variant=variant)
            flat[i] = old
            num = (lp - lm) / (2 * h)
            ana = grads[k].reshape(-1)[i]
            worst = max(worst, abs(num - ana) / max(1e-9, abs(num) + abs(ana)))
    return worst


def train_variant(variant, steps=8000, B=32, lr=2e-3, seed=0):
    print(f"\n{'='*70}")
    print(f"VARIANT {variant}")
    print(f"{'='*70}")

    err = grad_check(variant)
    print(f"gradient check: {err:.2e} ({'OK' if err < 1e-4 else 'FAIL'})")

    rng = np.random.default_rng(seed)
    p = init_params(rng, variant)
    adapter_params = sum(v.size for k, v in p.items()
                         if any(k.startswith(pre) for pre in
                                ["qbias_", "lora_", "out_scale_"]))
    total = sum(v.size for v in p.values())
    print(f"total params: {total:,}  (adapter: {adapter_params:,})")

    opt = Adam(p, lr=lr)
    eval_rng = np.random.default_rng(123)

    for step in range(1, steps + 1):
        x, y = get_batch("train", rng, B=B)
        loss, _, cache = forward(p, x, y, variant=variant)
        grads = backward(p, cache)
        opt.step(p, grads)
        if step % 500 == 0 or step == 1:
            tr_loss, tr_acc = evaluate(p, "train", eval_rng, variant)
            va_loss, va_acc = evaluate(p, "val", eval_rng, variant)
            print(f"step {step:4d} | train loss {tr_loss:6.3f} acc {tr_acc:5.1%}"
                  f" | val loss {va_loss:6.3f} acc {va_acc:5.1%}")

    np.savez(f"strat2_{variant}_params.npz", **p)
    print(f"saved -> strat2_{variant}_params.npz")
    return p


if __name__ == "__main__":
    for v in ["A", "B", "C"]:
        train_variant(v)
