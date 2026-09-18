"""
Step 3 — THE FULL GPT LOOP (forward + manual backprop + Adam)
============================================================
3-phase π-frequency attention with a ground wire.

Architecture:
    ids (B,T)
      -> token embeddings                     (B,T,C)
      -> LayerNorm
      -> 3-phase causal attention + ground    (B,T,C)
      -> + residual
      -> [ LayerNorm -> MLP (ReLU) -> + residual ]
      -> final LayerNorm
      -> linear head                          (B,T,vocab)
      -> cross-entropy loss vs the shifted targets y

The ground wire: a causal EMA of attention output that steers future
queries. Like the neutral wire in a 3-phase electrical system — it
carries memory of what attention has produced so far and biases where
attention goes next.

Two-pass attention:
  Pass 1: standard 3-phase attention (no ground) → raw attention output
  Ground: causal EMA of pass 1 output (position t sees 0..t-1)
  Pass 2: 3-phase attention with Q += α * ground → final output

Learned params: ground_alpha (injection strength), ground_decay_logit
(sigmoid → EMA decay rate). Total: 2 new scalars.

Run:  python step3_gpt.py          # trains and prints a learning curve
      python test_train.py         # asserts it actually learned
"""

import numpy as np
from step1_data import (
    train_data, val_data, vocab_size, BLOCK_SIZE, decode, stoi,
)

# ---- Hyperparameters ---------------------------------------------------------
C = 128         # embedding dim
N_PHASES = 3    # 3-phase attention (120° spacing)
MLP_HIDDEN = 4 * C
BLOCK = BLOCK_SIZE
EPS = 1e-5


# ============================================================================
# 3-phase π-frequency positional encoding
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
    ])  # (n_phases, T, C)


# ============================================================================
# Parameters
# ============================================================================
def init_params(rng, vocab=vocab_size, C=C):
    def n(*shape, scale):
        return rng.normal(0, scale, size=shape)
    return {
        "tok_emb": n(vocab, C, scale=0.02),
        # attention (shared across all 3 phases, both passes)
        "Wq": n(C, C, scale=1/np.sqrt(C)),
        "Wk": n(C, C, scale=1/np.sqrt(C)),
        "Wv": n(C, C, scale=1/np.sqrt(C)),
        "Wo": n(C, C, scale=1/np.sqrt(C)),
        # ground wire
        "ground_alpha": np.array(0.1),
        "ground_decay_logit": np.array(2.0),  # sigmoid → ~0.88
        # layernorms
        "g1": np.ones(C), "b1": np.zeros(C),
        "g2": np.ones(C), "b2": np.zeros(C),
        "gf": np.ones(C), "bf": np.zeros(C),
        # mlp
        "W1": n(C, MLP_HIDDEN, scale=1/np.sqrt(C)), "b1m": np.zeros(MLP_HIDDEN),
        "W2": n(MLP_HIDDEN, C, scale=1/np.sqrt(MLP_HIDDEN)), "b2m": np.zeros(C),
        # output head
        "Wh": n(C, vocab, scale=1/np.sqrt(C)), "bh": np.zeros(vocab),
    }


# ============================================================================
# Building blocks
# ============================================================================
def layernorm_fwd(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    istd = 1.0 / np.sqrt(var + EPS)
    xhat = (x - mu) * istd
    out = g * xhat + b
    return out, (xhat, istd, g)

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
# Single-pass 3-phase attention (used by both pass 1 and pass 2)
# ============================================================================
def _phase_attention_fwd(ln_tok, positions, p, q_bias=None):
    B, T, _ = ln_tok.shape
    n_phases = positions.shape[0]
    mask = np.tril(np.ones((T, T), dtype=bool))

    phase_outs = []
    phase_caches = []
    for phi in range(n_phases):
        x = ln_tok + positions[phi]
        q = x @ p["Wq"]
        if q_bias is not None:
            q = q + q_bias
        k = x @ p["Wk"]
        v = x @ p["Wv"]
        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(C)
        scores = np.where(mask, scores, -1e9)
        scores -= scores.max(-1, keepdims=True)
        e = np.exp(scores)
        att = e / e.sum(-1, keepdims=True)
        out = np.matmul(att, v)
        phase_outs.append(out)
        phase_caches.append((x, q, k, v, att, mask))

    combined = sum(phase_outs) / n_phases
    return combined, phase_caches


def _phase_attention_bwd(dcombined, phase_caches, p, has_q_bias=False):
    n_phases = len(phase_caches)
    dphase = dcombined / n_phases

    dWq = np.zeros_like(p["Wq"])
    dWk = np.zeros_like(p["Wk"])
    dWv = np.zeros_like(p["Wv"])
    dln_tok = np.zeros_like(phase_caches[0][0])  # same shape as x
    dq_bias_total = np.zeros_like(dln_tok) if has_q_bias else None

    for phi in range(n_phases):
        x, q, k, v, att, mask = phase_caches[phi]
        dv = np.matmul(att.transpose(0, 2, 1), dphase)
        datt = np.matmul(dphase, v.transpose(0, 2, 1))
        dscores = att * (datt - (datt * att).sum(-1, keepdims=True))
        dscores = np.where(mask, dscores, 0.0)
        dscores /= np.sqrt(C)
        dq = np.matmul(dscores, k)
        dk = np.matmul(dscores.transpose(0, 2, 1), q)

        dWq += np.einsum('btc,btd->cd', x, dq)
        dx = dq @ p["Wq"].T
        if has_q_bias:
            dq_bias_total += dq
        dWk += np.einsum('btc,btd->cd', x, dk)
        dWv += np.einsum('btc,btd->cd', x, dv)
        dx += dk @ p["Wk"].T + dv @ p["Wv"].T
        dln_tok += dx

    return dln_tok, dWq, dWk, dWv, dq_bias_total


# ============================================================================
# Ground wire attention (two-pass)
# ============================================================================
def attention_fwd(ln_tok, positions, p):
    B, T, _ = ln_tok.shape
    alpha = p["ground_alpha"]
    decay = 1.0 / (1.0 + np.exp(-p["ground_decay_logit"]))

    # Pass 1: standard 3-phase (no ground)
    combined_p1, p1_caches = _phase_attention_fwd(ln_tok, positions, p)

    # Build ground: causal EMA of pass 1 output
    ground = np.zeros((B, T, C))
    for t in range(1, T):
        ground[:, t] = decay * ground[:, t-1] + (1 - decay) * combined_p1[:, t-1]

    # Pass 2: ground-modulated Q
    q_bias = alpha * ground
    combined_p2, p2_caches = _phase_attention_fwd(ln_tok, positions, p, q_bias=q_bias)

    y = combined_p2 @ p["Wo"]
    cache = (ln_tok, positions, p1_caches, combined_p1, ground, p2_caches, combined_p2, decay)
    return y, cache


def attention_bwd(dy, cache, p):
    ln_tok, positions, p1_caches, combined_p1, ground, p2_caches, combined_p2, decay = cache
    alpha = p["ground_alpha"]
    B, T, _ = ln_tok.shape

    # Output projection
    dWo = np.einsum('btc,btd->cd', combined_p2, dy)
    dcombined_p2 = dy @ p["Wo"].T

    # Pass 2 backward
    dln_tok, dWq, dWk, dWv, dq_bias = _phase_attention_bwd(
        dcombined_p2, p2_caches, p, has_q_bias=True)

    # q_bias = alpha * ground
    dalpha = (dq_bias * ground).sum()
    dground = alpha * dq_bias

    # Ground EMA backward
    dcombined_p1 = np.zeros_like(combined_p1)
    ddecay = 0.0
    for t in range(T - 1, 0, -1):
        dcombined_p1[:, t-1] += (1 - decay) * dground[:, t]
        ddecay += ((ground[:, t-1] - combined_p1[:, t-1]) * dground[:, t]).sum()
        if t > 1:
            dground[:, t-1] += decay * dground[:, t]
    ddecay_logit = ddecay * decay * (1 - decay)

    # Pass 1 backward
    dln_tok_p1, dWq_p1, dWk_p1, dWv_p1, _ = _phase_attention_bwd(
        dcombined_p1, p1_caches, p, has_q_bias=False)
    dln_tok += dln_tok_p1
    dWq += dWq_p1
    dWk += dWk_p1
    dWv += dWv_p1

    return dln_tok, {
        "Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo,
        "ground_alpha": np.array(dalpha),
        "ground_decay_logit": np.array(ddecay_logit),
    }


# ============================================================================
# Full forward / backward
# ============================================================================
def forward(p, x_ids, y_ids=None):
    B, T = x_ids.shape
    tok = p["tok_emb"][x_ids]
    positions = multi_phase_pi_pos(T, C)

    ln1, c_ln1 = layernorm_fwd(tok, p["g1"], p["b1"])
    attn, c_att = attention_fwd(ln1, positions, p)
    h1 = tok + attn

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
    dmlp = dh2
    dact, grads["W2"], grads["b2m"] = linear_bwd(dmlp, cache["c_lin2"])
    dhid = dact * cache["relu_mask"]
    dln2, grads["W1"], grads["b1m"] = linear_bwd(dhid, cache["c_lin1"])
    dh1b, grads["g2"], grads["b2"] = layernorm_bwd(dln2, cache["c_ln2"])
    dh1 += dh1b

    dtok = dh1.copy()
    dattn = dh1
    dln1, gatt = attention_bwd(dattn, cache["c_att"], p)
    for kk, vv in gatt.items():
        grads[kk] += vv
    dtok_ln, grads["g1"], grads["b1"] = layernorm_bwd(dln1, cache["c_ln1"])
    dtok += dtok_ln

    np.add.at(grads["tok_emb"], cache["x_ids"], dtok)
    return grads


# ============================================================================
# Adam optimizer
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


# ============================================================================
# Batching + metrics
# ============================================================================
def get_batch(split, rng, B=32, T=BLOCK):
    src = train_data if split == "train" else val_data
    ix = rng.integers(0, len(src) - T, size=B)
    x = np.stack([src[i:i + T] for i in ix])
    y = np.stack([src[i + 1:i + 1 + T] for i in ix])
    return x, y

def evaluate(p, split, rng, B=64, T=BLOCK):
    x, y = get_batch(split, rng, B=B, T=T)
    loss, logits, _ = forward(p, x, y)
    pred = logits.argmax(-1)
    acc = (pred == y).mean()
    return loss, acc


# ============================================================================
# Training loop
# ============================================================================
def train(steps=1000, B=32, lr=3e-3, seed=0, log_every=100, verbose=True):
    rng = np.random.default_rng(seed)
    p = init_params(rng)
    opt = Adam(p, lr=lr)
    eval_rng = np.random.default_rng(123)
    history = []
    for step in range(1, steps + 1):
        x, y = get_batch("train", rng, B=B)
        loss, _, cache = forward(p, x, y)
        grads = backward(p, cache)
        opt.step(p, grads)
        if verbose and (step % log_every == 0 or step == 1):
            tr_loss, tr_acc = evaluate(p, "train", eval_rng)
            va_loss, va_acc = evaluate(p, "val", eval_rng)
            history.append((step, tr_loss, tr_acc, va_loss, va_acc))
            print(f"step {step:4d} | train loss {tr_loss:6.3f} acc {tr_acc:5.1%}"
                  f" | val loss {va_loss:6.3f} acc {va_acc:5.1%}")
    return p, history


def sample(p, n=200, seed=0, T=BLOCK, temperature=0.8):
    rng = np.random.default_rng(seed)
    out = [stoi["<eot>"]]
    for _ in range(n):
        x = np.array([out[-T:]])
        logits, _ = forward(p, x)
        z = logits[0, -1] / temperature
        probs = np.exp(z - z.max())
        probs /= probs.sum()
        nxt = int(rng.choice(len(probs), p=probs))
        out.append(nxt)
    return decode(out[1:])


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    p0 = init_params(np.random.default_rng(0))
    decay = 1.0 / (1.0 + np.exp(-p0["ground_decay_logit"]))
    print(f"vocab={vocab_size}  C={C}  phases={N_PHASES}  block(T)={BLOCK}  "
          f"params={sum(v.size for v in p0.values()):,}")
    print(f"ground: alpha={p0['ground_alpha']:.3f}  decay={decay:.3f}\n")
    p, hist = train(steps=1000, B=32, lr=3e-3)
    print(f"\nground after training: alpha={p['ground_alpha']:.4f}  "
          f"decay={1/(1+np.exp(-p['ground_decay_logit'])):.4f}")
    print("\n--- sample after training ---")
    print(sample(p, n=240))
