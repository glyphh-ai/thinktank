"""
Step 3 — THE FULL GPT LOOP (forward + manual backprop + Adam)
============================================================
Everything from steps 1-2 plus the rest of the transformer, with gradients
written out by hand (no autograd) so every number is traceable:

    ids (B,T)
      -> token embeddings                     (B,T,C)
      -> LayerNorm
      -> 3-phase causal attention (π-pos)     (B,T,C)
      -> + residual
      -> [ LayerNorm -> MLP (ReLU) -> + residual ]
      -> final LayerNorm
      -> linear head                          (B,T,vocab)
      -> cross-entropy loss vs the shifted targets y

3-phase attention: same Wq/Wk/Wv/Wo weights process the input three times,
each with a different phase of the π-frequency positional encoding (0°, 120°,
240°). Like a 3-phase electrical transformer — no positional blind spots,
every position is strongly visible to at least one phase at every frequency.

Run:  python step3_gpt.py          # trains and prints a learning curve
      python test_train.py         # asserts it actually learned
"""

import numpy as np
from step1_data import (
    train_data, val_data, vocab_size, BLOCK_SIZE, decode, stoi,
)

# ---- Hyperparameters ---------------------------------------------------------
C = 128         # embedding dim (n_embd) — widened for the word-level head
N_PHASES = 3    # 3-phase attention (120° spacing)
MLP_HIDDEN = 4 * C
BLOCK = BLOCK_SIZE  # T (context length)
EPS = 1e-5      # layernorm epsilon


# ============================================================================
# 3-phase π-frequency positional encoding
# ============================================================================
def multi_phase_pi_pos(T, C, n_phases=N_PHASES):
    """N phase-shifted π-frequency positional encodings, equally spaced.

    Each phase is a full (T, C) sinusoidal encoding at base frequency π,
    offset by 2πk/N. Like the windings of a polyphase transformer —
    together they cover the full rotation, eliminating positional blind spots.
    """
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
    """All learnable tensors, small random init (scaled so signals stay sane)."""
    def n(*shape, scale):
        return rng.normal(0, scale, size=shape)
    return {
        "tok_emb": n(vocab, C, scale=0.02),
        # attention (shared across all 3 phases)
        "Wq": n(C, C, scale=1/np.sqrt(C)),
        "Wk": n(C, C, scale=1/np.sqrt(C)),
        "Wv": n(C, C, scale=1/np.sqrt(C)),
        "Wo": n(C, C, scale=1/np.sqrt(C)),
        # layernorms (gain g, bias b)
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
# Small differentiable building blocks (each returns forward value + a cache)
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
# 3-phase causal self-attention
# ============================================================================
def attention_fwd(ln_tok, positions, p):
    """3-phase attention: shared weights, 3 phase-shifted position inputs.

    ln_tok:     (B, T, C)  — layernormed token embeddings (no position)
    positions:  (3, T, C)  — three π-positional encodings at 0°, 120°, 240°

    Each phase adds its positional encoding to ln_tok, computes Q/K/V through
    the same weights, runs causal attention, and produces an output. The three
    outputs are averaged and projected through Wo.
    """
    B, T, _ = ln_tok.shape
    n_phases = positions.shape[0]
    mask = np.tril(np.ones((T, T), dtype=bool))

    phase_outs = []
    phase_caches = []
    for phi in range(n_phases):
        x = ln_tok + positions[phi]               # (B,T,C)
        q = x @ p["Wq"]                           # (B,T,C)
        k = x @ p["Wk"]
        v = x @ p["Wv"]
        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(C)  # (B,T,T)
        scores = np.where(mask, scores, -1e9)
        scores -= scores.max(-1, keepdims=True)
        e = np.exp(scores)
        att = e / e.sum(-1, keepdims=True)         # (B,T,T)
        out = np.matmul(att, v)                    # (B,T,C)
        phase_outs.append(out)
        phase_caches.append((x, q, k, v, att, out, mask))

    combined = sum(phase_outs) / n_phases          # (B,T,C)
    y = combined @ p["Wo"]                         # (B,T,C)
    return y, (ln_tok, positions, phase_caches, combined)


def attention_bwd(dy, cache, p):
    ln_tok, positions, phase_caches, combined = cache
    n_phases = len(phase_caches)

    dWo = np.einsum('btc,btd->cd', combined, dy)
    dcombined = dy @ p["Wo"].T                     # (B,T,C)
    dphase = dcombined / n_phases

    dWq = np.zeros_like(p["Wq"])
    dWk = np.zeros_like(p["Wk"])
    dWv = np.zeros_like(p["Wv"])
    dln_tok = np.zeros_like(ln_tok)

    for phi in range(n_phases):
        x, q, k, v, att, out, mask = phase_caches[phi]

        # out = att @ v
        dv = np.matmul(att.transpose(0, 2, 1), dphase)
        datt = np.matmul(dphase, v.transpose(0, 2, 1))

        # softmax backward
        dscores = att * (datt - (datt * att).sum(-1, keepdims=True))
        dscores = np.where(mask, dscores, 0.0)
        dscores /= np.sqrt(C)

        # scores = q @ k^T
        dq = np.matmul(dscores, k)
        dk = np.matmul(dscores.transpose(0, 2, 1), q)

        # q,k,v = x @ Wq,Wk,Wv — accumulate across phases
        dWq += np.einsum('btc,btd->cd', x, dq)
        dWk += np.einsum('btc,btd->cd', x, dk)
        dWv += np.einsum('btc,btd->cd', x, dv)

        # dx -> dln_tok (positions are fixed, gradient flows only to token input)
        dx = dq @ p["Wq"].T + dk @ p["Wk"].T + dv @ p["Wv"].T
        dln_tok += dx

    return dln_tok, {"Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo}


# ============================================================================
# Full forward (returns loss, predictions, and a cache for backward)
# ============================================================================
def forward(p, x_ids, y_ids=None):
    B, T = x_ids.shape
    tok = p["tok_emb"][x_ids]                      # (B,T,C)
    positions = multi_phase_pi_pos(T, C)           # (3,T,C) — deterministic

    ln1, c_ln1 = layernorm_fwd(tok, p["g1"], p["b1"])
    attn, c_att = attention_fwd(ln1, positions, p)
    h1 = tok + attn                                # residual to token embeddings

    ln2, c_ln2 = layernorm_fwd(h1, p["g2"], p["b2"])
    hid, c_lin1 = linear_fwd(ln2, p["W1"], p["b1m"])
    relu_mask = hid > 0
    act = hid * relu_mask
    mlp, c_lin2 = linear_fwd(act, p["W2"], p["b2m"])
    h2 = h1 + mlp

    lnf, c_lnf = layernorm_fwd(h2, p["gf"], p["bf"])
    logits, c_head = linear_fwd(lnf, p["Wh"], p["bh"])   # (B,T,V)

    cache = dict(x_ids=x_ids, B=B, T=T,
                 c_ln1=c_ln1, c_att=c_att, c_ln2=c_ln2,
                 c_lin1=c_lin1, relu_mask=relu_mask, c_lin2=c_lin2,
                 c_lnf=c_lnf, c_head=c_head)

    if y_ids is None:
        return logits, cache

    # cross-entropy
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

    # d loss / d logits
    dlogits = probs.copy()
    dlogits[np.arange(N), yf] -= 1.0
    dlogits /= N
    dlogits = dlogits.reshape(B, T, vocab_size)

    # head
    dlnf, grads["Wh"], grads["bh"] = linear_bwd(dlogits, cache["c_head"])
    dh2, grads["gf"], grads["bf"] = layernorm_bwd(dlnf, cache["c_lnf"])

    # mlp residual: h2 = h1 + mlp
    dh1 = dh2.copy()
    dmlp = dh2
    dact, grads["W2"], grads["b2m"] = linear_bwd(dmlp, cache["c_lin2"])
    dhid = dact * cache["relu_mask"]
    dln2, grads["W1"], grads["b1m"] = linear_bwd(dhid, cache["c_lin1"])
    dh1b, grads["g2"], grads["b2"] = layernorm_bwd(dln2, cache["c_ln2"])
    dh1 += dh1b

    # attention residual: h1 = tok + attn
    dtok = dh1.copy()
    dattn = dh1
    dln1, gatt = attention_bwd(dattn, cache["c_att"], p)
    for kk, vv in gatt.items():
        grads[kk] += vv
    dtok_ln, grads["g1"], grads["b1"] = layernorm_bwd(dln1, cache["c_ln1"])
    dtok += dtok_ln

    # token embeddings (positions are fixed — no gradient)
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
    """Return (loss, accuracy) on one freshly-drawn batch."""
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
    """Autoregressively generate text to eyeball what it learned.

    Starts at a story boundary (<eot>) so generation opens like a story.
    """
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
    print(f"vocab={vocab_size}  C={C}  phases={N_PHASES}  block(T)={BLOCK}  "
          f"params={sum(v.size for v in init_params(np.random.default_rng(0)).values())}\n")
    p, hist = train(steps=1000, B=32, lr=3e-3)
    print("\n--- sample after training ---")
    print(sample(p, n=240))
