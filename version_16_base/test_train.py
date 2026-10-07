"""
Tests for the base GPT.

  1. gradient check — every hand-written gradient vs central finite
     differences (float64, a few random entries per tensor)
  2. it learns      — 150 steps must clearly beat the uniform-guess loss

Run:  python test_train.py
"""

import numpy as np

import step3_gpt as G


def test_gradients():
    rng = np.random.default_rng(0)
    p = {k: v.astype(np.float64) for k, v in G.init_params(rng).items()}
    # larger-than-init weights so every path carries a real signal
    for k in p:
        if p[k].ndim >= 2:
            p[k] *= 5
        else:
            p[k] += rng.normal(0, 0.1, p[k].shape)
    x = rng.integers(0, G.vocab_size, (2, 8))
    y = rng.integers(0, G.vocab_size, (2, 8))
    _, _, cache = G.forward(p, x, y)
    grads = G.backward(p, cache)

    worst = 0.0
    for k, w in p.items():
        flat = w.reshape(-1)
        if k == "tok_emb":   # only rows that are used (input or target) matter
            rows = np.unique(np.concatenate([x.ravel(), y.ravel()]))
            idx = [r * G.C + c for r in rows[:4] for c in rng.integers(0, G.C, 2)]
        elif k == "pos_emb":
            idx = rng.integers(0, 8 * G.C, 6)
        else:
            idx = rng.integers(0, flat.size, 6)
        for i in idx:
            old = flat[i]
            flat[i] = old + 1e-5; lp = G.forward(p, x, y)[0]
            flat[i] = old - 1e-5; lm = G.forward(p, x, y)[0]
            flat[i] = old
            num = (lp - lm) / 2e-5
            ana = grads[k].reshape(-1)[i]
            rel = abs(num - ana) / max(1e-6, abs(num) + abs(ana))
            worst = max(worst, rel)
            assert rel < 1e-3, f"{k}[{i}] numeric {num:.6g} vs analytic {ana:.6g}"
    print(f"gradient check: PASS (worst relative error {worst:.1e})")


def test_learns():
    uniform = np.log(G.vocab_size)
    p, hist = G.train(steps=150, B=16, lr=2e-3, log_every=150, verbose=True)
    val_loss = hist[-1][3]
    assert val_loss < uniform - 3.0, f"val loss {val_loss:.3f} vs uniform {uniform:.3f}"
    print(f"it learns: PASS (val loss {val_loss:.3f}, uniform {uniform:.3f})")


if __name__ == "__main__":
    test_gradients()
    test_learns()
