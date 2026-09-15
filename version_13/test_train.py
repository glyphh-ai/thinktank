"""
Benchmark / regression test for the step-3 GPT.

Runs the full loop for 500-1000 Adam steps and asserts the model actually
learned: loss must fall a lot and next-word accuracy must clear a threshold.
Also re-checks backprop with finite differences so a bad gradient can never
masquerade as "it trained fine".

Run:   python test_train.py          # prints a benchmark table, exits non-zero on fail
   or: pytest test_train.py -q
"""

import numpy as np
import step3_gpt as G


# ----------------------------------------------------------------------------
def _grad_check_max_error(seed=1, n_per_param=6, h=1e-5):
    rng = np.random.default_rng(seed)
    p = G.init_params(rng)
    x = np.array([[3, 8, 1, 8, 2], [5, 16, 7, 1, 6]])
    y = np.array([[8, 1, 8, 2, 1], [16, 7, 1, 6, 25]])
    _, _, cache = G.forward(p, x, y)
    grads = G.backward(p, cache)
    worst = 0.0
    for k in p:
        flat = p[k].reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_per_param, flat.size), replace=False)
        for i in idxs:
            old = flat[i]
            flat[i] = old + h; lp, _, _ = G.forward(p, x, y)
            flat[i] = old - h; lm, _, _ = G.forward(p, x, y)
            flat[i] = old
            num = (lp - lm) / (2 * h)
            ana = grads[k].reshape(-1)[i]
            worst = max(worst, abs(num - ana) / max(1e-9, abs(num) + abs(ana)))
    return worst


def _benchmark(steps=700, B=32, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    p = G.init_params(rng)
    opt = G.Adam(p, lr=lr)
    eval_rng = np.random.default_rng(123)

    init_loss, init_acc = G.evaluate(p, "val", eval_rng)
    for step in range(1, steps + 1):
        x, y = G.get_batch("train", rng, B=B)
        loss, _, cache = G.forward(p, x, y)
        opt.step(p, G.backward(p, cache))

    # average several batches so the numbers aren't a lucky/unlucky draw
    tr = [G.evaluate(p, "train", eval_rng) for _ in range(8)]
    va = [G.evaluate(p, "val", eval_rng) for _ in range(8)]
    tr_loss = np.mean([l for l, _ in tr]); tr_acc = np.mean([a for _, a in tr])
    va_loss = np.mean([l for l, _ in va]); va_acc = np.mean([a for _, a in va])
    return dict(init_loss=init_loss, init_acc=init_acc,
                tr_loss=tr_loss, tr_acc=tr_acc,
                va_loss=va_loss, va_acc=va_acc, steps=steps)


# ----------------------------------------------------------------------------
# pytest entry points
# ----------------------------------------------------------------------------
def test_gradients_match_numeric():
    err = _grad_check_max_error()
    assert err < 1e-4, f"backprop disagrees with finite differences: {err:.2e}"


def test_model_learns():
    r = _benchmark(steps=700)
    # word-level bars (vocab ~5.4k, init loss ~ ln(V) = 8.6): loss must
    # roughly halve, and accuracy must clearly beat the always-predict-'.'
    # baseline (~8%). Measured at 700 steps: val loss ~3.9, val acc ~27%.
    assert r["va_loss"] < 0.55 * r["init_loss"], r
    assert r["va_loss"] < 4.3, r
    assert r["tr_acc"] > 0.22, r
    assert r["va_acc"] > 0.22, r
    # train/val gap small -> not wildly overfitting
    assert abs(r["tr_acc"] - r["va_acc"]) < 0.08, r


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("== gradient check ==")
    err = _grad_check_max_error()
    print(f"max relative error vs finite differences: {err:.2e}  "
          f"({'OK' if err < 1e-4 else 'FAIL'})\n")

    print("== training benchmark ==")
    r = _benchmark(steps=700)
    print(f"baseline (step 0)   : loss {r['init_loss']:6.3f}  acc {r['init_acc']:6.1%}")
    print(f"after {r['steps']} steps :")
    print(f"  train             : loss {r['tr_loss']:6.3f}  acc {r['tr_acc']:6.1%}")
    print(f"  val               : loss {r['va_loss']:6.3f}  acc {r['va_acc']:6.1%}")

    ok = (err < 1e-4 and r["va_loss"] < 4.3
          and r["tr_acc"] > 0.22 and r["va_acc"] > 0.22)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
