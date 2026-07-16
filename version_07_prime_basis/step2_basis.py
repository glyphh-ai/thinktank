"""step2 — THE test: are the NSM primes a privileged linear basis?

For each of 100 target words, reconstruct its concept vector as a sparse
linear combination (Orthogonal Matching Pursuit, k nonzero coefficients) of:

  (a) the 61 prime vectors                      — the hypothesis
  (b) 61 random words from the frequency-matched control pool — the null,
      resampled N_NULL times

Statistic: mean reconstruction R^2 over targets. Empirical p-value = fraction
of null dictionaries whose mean R^2 >= the primes'. If the primes don't beat
frequency-matched random dictionaries, the hypothesis is DEAD.

Signed coefficients are allowed (NSM explications use negation: "bad" involves
NOT + GOOD), sparsity is fixed and identical across dictionaries so the
comparison is fair. Run at k = 4, 8, 16 for robustness.

Layer selection is NOT fit to this test: the layer is chosen once, upstream,
to maximize split-half reliability over ALL words (a measurement criterion,
blind to the prime/control distinction).
"""
import argparse

import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit

from concepts import CONTROL_POOL, PRIMES, TARGETS

N_NULL = 20
KS = (4, 8, 16)


def load_vectors(path="vectors.npz", split="both"):
    """Return dict word -> centered, unit-norm vector at the most reliable layer.

    split: 'both' (halves A+B pooled), 'usage', or 'mention'.
    Also returns (layer, reliability) metadata.
    """
    z = np.load(path, allow_pickle=True)
    sums, counts = z["sums"], z["counts"]
    words = [str(w) for w in z["words"]]
    splits = [str(s) for s in z["split_names"]]
    means = sums / counts[None, :, None, None]  # [W, S, L, d]

    ia, ib = splits.index("half_a"), splits.index("half_b")
    # layer choice: maximize median split-half cosine over all words (after centering)
    rel_by_layer = []
    for layer in range(means.shape[2]):
        a = means[:, ia, layer, :] - means[:, ia, layer, :].mean(0)
        b = means[:, ib, layer, :] - means[:, ib, layer, :].mean(0)
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        rel_by_layer.append(np.median(cos))
    layer = int(np.argmax(rel_by_layer))
    reliability = float(rel_by_layer[layer])

    if split == "both":
        v = (means[:, ia, layer, :] + means[:, ib, layer, :]) / 2
    else:
        v = means[:, splits.index(split), layer, :]
    v = v - v.mean(0)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return {w: v[i] for i, w in enumerate(words)}, layer, reliability, rel_by_layer


def omp_r2(dictionary, target, k):
    """R^2 of reconstructing `target` (d,) from rows of `dictionary` (n, d)."""
    X = dictionary.T  # (d, n)
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=False)
    omp.fit(X, target)
    resid = target - X @ omp.coef_
    return 1.0 - (resid @ resid) / (target @ target), omp.coef_


def run_basis_test(vecs, rng, ks=KS, n_null=N_NULL, verbose=True):
    D_prime = np.stack([vecs[w] for w in PRIMES])
    pool = list(CONTROL_POOL)
    targets = {w: vecs[w] for w in TARGETS}

    results = {}
    for k in ks:
        prime_r2, usage_count = [], np.zeros(len(PRIMES))
        for w, t in targets.items():
            r2, coef = omp_r2(D_prime, t, k)
            prime_r2.append(r2)
            usage_count += (coef != 0)
        prime_mean = float(np.mean(prime_r2))

        null_means, null_all = [], []
        for _ in range(n_null):
            idx = rng.choice(len(pool), size=len(PRIMES), replace=False)
            D_null = np.stack([vecs[pool[i]] for i in idx])
            r2s = [omp_r2(D_null, t, k)[0] for t in targets.values()]
            null_means.append(float(np.mean(r2s)))
            null_all.append(r2s)
        null_means = np.array(null_means)
        p = float((null_means >= prime_mean).mean())
        # per-target win rate vs the null-average for that target
        null_per_target = np.mean(np.array(null_all), axis=0)
        win = float(np.mean(np.array(prime_r2) > null_per_target))

        results[k] = dict(prime_mean=prime_mean, null_mean=float(null_means.mean()),
                          null_std=float(null_means.std()), p=p, win_rate=win,
                          prime_r2=prime_r2, usage=usage_count)
        if verbose:
            print(f"k={k:2d}  primes R2={prime_mean:.4f}  "
                  f"null R2={null_means.mean():.4f} +/- {null_means.std():.4f}  "
                  f"p={p:.3f}  per-target win rate={win:.2f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="vectors.npz")
    ap.add_argument("--split", default="both", choices=["both", "usage", "mention"])
    args = ap.parse_args()

    vecs, layer, rel, _ = load_vectors(args.vectors, args.split)
    print(f"layer {layer} (split-half reliability, median cos = {rel:.3f}), "
          f"split={args.split}\n")

    rng = np.random.default_rng(7)
    results = run_basis_test(vecs, rng)

    usage = results[8]["usage"]
    order = np.argsort(-usage)
    print("\nmost-used primes (k=8):")
    for i in order[:12]:
        print(f"  {PRIMES[i]:<10s} used in {int(usage[i])}/100 targets")
    print("never used:", ", ".join(PRIMES[i] for i in range(len(PRIMES)) if usage[i] == 0) or "(none)")


if __name__ == "__main__":
    main()
