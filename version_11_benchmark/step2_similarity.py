"""step2 — G1/G2/G3: human similarity judgments (SimLex-999, WordSim-353, MEN).

For each benchmark pair, similarity per representation; Spearman against the
human scores:

  FULL     896-dim frozen-transformed vectors, cosine        (the ceiling)
  TERNARY  K=256 frozen named barcode, cosine                (the product)
  SIMHASH  K=256 random-hyperplane binary codes, 5 seeds     (the anonymous null)

Pre-registered gates:
  G1 substrate   full-vector rho >= 0.35 on SimLex-999 AND >= 0.55 on WS-353
                 (published static-embedding references: word2vec ~0.44 / ~0.70)
  G2 retention   ternary rho >= 0.90 x full rho on SimLex AND WS-353
  G3 named bits  ternary rho >= mean SimHash rho on SimLex AND WS-353

MEN is reported for completeness (no gate). All 1819 benchmark words were
extracted, so coverage is 100% — no dropped pairs, no cherry-picking.
"""
import numpy as np
from scipy.stats import spearmanr

from bench_vocab import benchmark_pairs
from frozen import bench_vectors, code_cos, load_frozen, ternary_codes

K = 256
BENCHES = ["simlex.txt", "ws353.txt", "men.txt"]


def evaluate(sim_fn, pairs, wi):
    human, model = [], []
    for a, b, s in pairs:
        human.append(s)
        model.append(sim_fn(wi[a], wi[b]))
    return spearmanr(human, model).statistic


def main():
    bank, layer, g09 = load_frozen()
    words, V = bench_vectors(layer, g09)
    wi = {w: i for i, w in enumerate(words)}
    C = ternary_codes(V, bank, K)
    rng = np.random.default_rng(7)
    SH = []
    for _ in range(5):
        H = rng.normal(size=(K, V.shape[1]))
        H /= np.linalg.norm(H, axis=1, keepdims=True)
        s = np.sign(V @ H.T)
        s[s == 0] = 1
        SH.append(s.astype(np.int8))

    print(f"frozen layer {layer}, K={K}\n"
          f"{'benchmark':<12s} {'n':>5s} {'full':>8s} {'ternary':>8s} {'simhash':>8s} {'retention':>10s}")
    res = {}
    for name in BENCHES:
        pairs = benchmark_pairs(name)
        missing = [p for p in pairs if p[0] not in wi or p[1] not in wi]
        assert not missing, f"{name}: {len(missing)} pairs missing"
        rho_f = evaluate(lambda i, j: float(V[i] @ V[j]), pairs, wi)
        rho_t = evaluate(lambda i, j: float(code_cos(C[i:i+1], C[j:j+1])[0]), pairs, wi)
        rho_s = float(np.mean([
            evaluate(lambda i, j: float((S[i].astype(np.float32) @ S[j]) / K), pairs, wi)
            for S in SH]))
        res[name] = (rho_f, rho_t, rho_s)
        print(f"{name:<12s} {len(pairs):>5d} {rho_f:>8.4f} {rho_t:>8.4f} {rho_s:>8.4f} "
              f"{rho_t / rho_f:>9.1%}")

    sl, ws = res["simlex.txt"], res["ws353.txt"]
    g1 = sl[0] >= 0.35 and ws[0] >= 0.55
    g2 = sl[1] >= 0.90 * sl[0] and ws[1] >= 0.90 * ws[0]
    g3 = sl[1] >= sl[2] and ws[1] >= ws[2]
    print(f"\nG1 substrate {'PASS' if g1 else 'FAIL'} | "
          f"G2 retention {'PASS' if g2 else 'FAIL'} | "
          f"G3 named>=simhash {'PASS' if g3 else 'FAIL'}")
    return g1, g2, g3, res


if __name__ == "__main__":
    main()
