"""EXPLORATORY (not pre-registered): T3 failed at the pre-registered D=8192
on recall alone (0.925 vs 0.95). This sweep asks whether the failure is a
design flaw or a capacity constant: same codes, same read rule, D doubled.

Result: D=16384 -> recovery 0.9996, query precision & recall 1.000.
The pre-registered T3 verdict stands as FAIL at D=8192; the carrier design
is sound and the production constant is D >= 16k for K=256 mostly-committed
ternary bits.
"""
import numpy as np

from ternary import eval_vectors, load_bank, ternary_codes

K = 256


def main():
    bank = load_bank()
    names = list(bank["names"][:K])
    words, V = eval_vectors(int(bank["layer"]))
    C = ternary_codes(V, bank, K)
    vi, ai = names.index("good-bad"), names.index("someone-something")
    truth = (C[:, vi] == -1) & (C[:, ai] == +1)

    for D in (8192, 16384, 32768):
        rng = np.random.default_rng(5)
        ids = (rng.integers(0, 2, size=(K, D)) * 2 - 1).astype(np.int8)
        rec = np.sign(C.astype(np.int32) @ ids.astype(np.int32))
        rec[rec == 0] = 1
        n_w = np.abs(C).sum(1)
        s = (rec @ ids.T.astype(np.int32)).astype(np.float64) / D
        theta = 0.5 * np.sqrt(2 / (np.pi * np.maximum(n_w, 1)))[:, None]
        read = np.where(np.abs(s) < theta, 0, np.sign(s)).astype(np.int8)
        acc = (read == C).mean()
        pred = (read[:, vi] == -1) & (read[:, ai] == +1)
        tp = (pred & truth).sum()
        prec, recall = tp / max(pred.sum(), 1), tp / max(truth.sum(), 1)
        print(f"D={D:>6d}: recovery {acc:.4f}, query P {prec:.3f} R {recall:.3f}")


if __name__ == "__main__":
    main()
