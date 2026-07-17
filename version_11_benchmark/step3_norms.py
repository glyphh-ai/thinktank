"""step3 — G4/G5: human valence norms (Warriner et al. 2013, 1-9 ratings).

G4 ZERO-SHOT: the frozen good-bad axis reading (one z-score per word, no
training whatsoever) against human valence for every extracted word with a
Warriner rating (~2.9k). Gate: Spearman >= 0.50. The sign convention is
fixed by the axis name (good side = positive) — nothing is fit.

G5 FEATURES: out-of-fold ridge predictions (5-fold, alpha CV) of human
valence from (a) full 896-dim vectors, (b) the 256 frozen ternary bits,
(c) SimHash-256 bits. Gate: ternary >= 0.90 x full AND ternary > simhash.
Tests whether 256 named bits are as good a *feature set* as the raw
embedding they were distilled from.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from bench_vocab import warriner
from frozen import bench_vectors, load_frozen, ternary_codes, zscores

K = 256


def oof_ridge(X, y, seed=0):
    pred = np.zeros_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return spearmanr(y, pred).statistic


def main():
    bank, layer, g09 = load_frozen()
    words, V = bench_vectors(layer, g09)
    val = warriner()
    idx = [i for i, w in enumerate(words) if w in val]
    y = np.array([val[words[i]] for i in idx])
    print(f"frozen layer {layer}: {len(idx)} extracted words have Warriner valence")

    names = list(bank["names"])
    gb = names.index("good-bad")
    z_gb = zscores(V[idx], bank)[:, gb]
    rho_zero = spearmanr(y, z_gb).statistic
    print(f"G4 zero-shot good-bad axis vs human valence: rho = {rho_zero:.4f} "
          f"(gate >= 0.50) -> {'PASS' if rho_zero >= 0.50 else 'FAIL'}")

    Xf = V[idx]
    Xt = ternary_codes(V[idx], bank, K).astype(np.float32)
    rng = np.random.default_rng(7)
    H = rng.normal(size=(K, V.shape[1]))
    H /= np.linalg.norm(H, axis=1, keepdims=True)
    Xs = np.sign(V[idx] @ H.T).astype(np.float32)

    rho_f, rho_t, rho_s = (oof_ridge(X, y) for X in (Xf, Xt, Xs))
    print(f"G5 ridge (out-of-fold Spearman): full {rho_f:.4f} | "
          f"ternary-256 {rho_t:.4f} ({rho_t / rho_f:.1%}) | simhash-256 {rho_s:.4f}")
    g5 = rho_t >= 0.90 * rho_f and rho_t > rho_s
    print(f"G5 {'PASS' if g5 else 'FAIL'}")
    return rho_zero >= 0.50, g5


if __name__ == "__main__":
    main()
