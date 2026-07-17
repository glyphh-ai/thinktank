"""falsify.py — version_12 verdict: the same gates v11 failed, wide seeds.

Identical bars to v11, re-graded at layer 12 with the 242-pair bank:

  G1 SUBSTRATE  full vectors at layer 12: SimLex >= 0.35, WS-353 >= 0.55
  G2 RETENTION  ternary-256 >= 0.90 x full rho on SimLex AND WS-353
  G3 NAMED      ternary-256 >= mean SimHash-256 (5 seeds) on both
  G4 ZERO-SHOT  good-bad z vs Warriner valence rho >= 0.50 (disclosure:
                layer 12 was picked partly on Warriner in v11's sweep)
  G5 FEATURES   ridge on ternary-256 >= 0.90 x full ridge AND > simhash

Extras reported ungated: K=512 retention (the wide bank allows it), and a
no-overlap robustness split for G2/G3 (benchmark pairs with either word in
the seed inventory excluded).
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "version_11_benchmark"))
from bench_vocab import benchmark_pairs, warriner  # noqa: E402

from pairs import seed_words  # noqa: E402
from wideband import (bench_vectors, build_bank, code_cos_rows, load_bank,  # noqa: E402
                      ternary_codes)

K = 256


def rho_pairs(pairs, sim):
    return spearmanr([s for _, _, s in pairs], [sim(a, b) for a, b, _ in pairs]).statistic


def main():
    n_kept, n_total = build_bank()
    bank = load_bank()
    words, V = bench_vectors(bank)
    wi = {w: i for i, w in enumerate(words)}
    print(f"bank: {n_kept}/{n_total} axes certified at layer {int(bank['layer'])}")

    C = ternary_codes(V, bank, K)
    rng = np.random.default_rng(7)
    SH = []
    for _ in range(5):
        H = rng.normal(size=(K, V.shape[1]))
        H /= np.linalg.norm(H, axis=1, keepdims=True)
        s = np.sign(V @ H.T)
        s[s == 0] = 1
        SH.append(s.astype(np.int8))

    seeds = set(seed_words())
    res = {}
    print(f"\n{'benchmark':<12s} {'full':>8s} {'tern-256':>9s} {'simhash':>8s} "
          f"{'retention':>10s} {'no-overlap ret.':>16s}")
    for name in ("simlex.txt", "ws353.txt", "men.txt"):
        pairs = benchmark_pairs(name)
        f = rho_pairs(pairs, lambda a, b: float(V[wi[a]] @ V[wi[b]]))
        t = rho_pairs(pairs, lambda a, b: code_cos_rows(C, wi[a], wi[b]))
        s = float(np.mean([rho_pairs(pairs, lambda a, b, S=S: float(
            (S[wi[a]].astype(np.float32) @ S[wi[b]]) / K)) for S in SH]))
        clean = [p for p in pairs if p[0] not in seeds and p[1] not in seeds]
        fc = rho_pairs(clean, lambda a, b: float(V[wi[a]] @ V[wi[b]]))
        tc = rho_pairs(clean, lambda a, b: code_cos_rows(C, wi[a], wi[b]))
        res[name] = (f, t, s)
        print(f"{name:<12s} {f:>8.4f} {t:>9.4f} {s:>8.4f} {t / f:>9.1%} "
              f"{tc / fc:>10.1%} (n={len(clean)})")

    sl, ws = res["simlex.txt"], res["ws353.txt"]
    g1 = sl[0] >= 0.35 and ws[0] >= 0.55
    g2 = sl[1] >= 0.90 * sl[0] and ws[1] >= 0.90 * ws[0]
    g3 = sl[1] >= sl[2] and ws[1] >= ws[2]

    # K=512 report
    if len(bank["axes"]) >= 512:
        C512 = ternary_codes(V, bank, 512)
        t512 = {n: rho_pairs(benchmark_pairs(n),
                             lambda a, b: code_cos_rows(C512, wi[a], wi[b]))
                for n in ("simlex.txt", "ws353.txt")}
        print(f"K=512 (report): SimLex {t512['simlex.txt']:.4f} "
              f"({t512['simlex.txt'] / sl[0]:.1%}), WS353 {t512['ws353.txt']:.4f} "
              f"({t512['ws353.txt'] / ws[0]:.1%})")

    # G4/G5
    val = warriner()
    idx = [i for i, w in enumerate(words) if w in val]
    y = np.array([val[words[i]] for i in idx])
    names = list(bank["names"])
    gb = names.index("good-bad")
    zgb = ((V[idx] @ bank["axes"][gb]) - bank["mu"][gb]) / bank["sd"][gb]
    g4rho = spearmanr(y, zgb).statistic
    g4 = g4rho >= 0.50
    print(f"\nG4 zero-shot good-bad vs valence (n={len(idx)}): rho={g4rho:.4f} "
          f"-> {'PASS' if g4 else 'FAIL'}")

    def oof(X):
        pred = np.zeros_like(y)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        return spearmanr(y, pred).statistic

    rf, rt, rs = oof(V[idx]), oof(C[idx].astype(np.float32)), oof(SH[0][idx].astype(np.float32))
    g5 = rt >= 0.90 * rf and rt > rs
    print(f"G5 ridge: full {rf:.4f} | ternary {rt:.4f} ({rt / rf:.1%}) | "
          f"simhash {rs:.4f} -> {'PASS' if g5 else 'FAIL'}")

    print(f"\nG1 {'PASS' if g1 else 'FAIL'} | G2 {'PASS' if g2 else 'FAIL'} | "
          f"G3 {'PASS' if g3 else 'FAIL'} | G4 {'PASS' if g4 else 'FAIL'} | "
          f"G5 {'PASS' if g5 else 'FAIL'}")
    if g1 and g2 and g3 and g4 and g5:
        print("VERDICT: LEGIT — the wide-seed named barcode passes the external "
              "gates v11 failed.")
    elif not g1:
        print("VERDICT: SUBSTRATE WEAK at layer 12.")
    else:
        print("VERDICT: PARTIAL — see gate flags.")


if __name__ == "__main__":
    main()
