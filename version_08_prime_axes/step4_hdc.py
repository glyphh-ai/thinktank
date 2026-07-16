"""step4 — H5: can the Glyph HDC lattice carry an axis profile? bind vs bundle.

The v06 bridge. A target word's 13-axis profile is encoded as one bipolar
hypervector (D=8192) two ways:

  BOUND   bundle_i( bind(axis_id_i, level(q_i)) )   — role x filler, the
          lattice's native encoding: which axis had which value is preserved.
  UNBOUND bundle_i( level(q_i) )                    — same information
          bundled WITHOUT axis identity: a bag of levels.

Levels are a thermometer code (chained random sign-flips), so nearby
projection values get similar level vectors. Test: Spearman rho between
HDC-space cosine similarity and 13-dim profile cosine similarity, over all
target pairs.

Pre-registered pass for H5: bound rho >= 0.6 AND bound >= unbound + 0.15.
This is a capacity/faithfulness test of the encoding, not a semantics
discovery — it decides whether prime-axis profiles can live in the
glyphh-runtime lattice without geometric loss.

HDC ops vendored from version_06_hdc_forward/hdc.py conventions:
bind = elementwise multiply, bundle = majority sign, bipolar int8.
"""
import numpy as np
from scipy.stats import spearmanr

from axes import AXES, AXIS_NAMES
from step1_project import load_all, project
from step2_geometry import sim_upper, zscore

D = 8192
N_LEVELS = 7
FLIP_FRAC = 0.15  # per level step; adjacent-level cos ~ 0.7
BIN_EDGES = [-1.5, -0.75, -0.25, 0.25, 0.75, 1.5]  # z-score bins -> 7 levels


def rand_bipolar(rng, n=1):
    return (rng.integers(0, 2, size=(n, D)) * 2 - 1).astype(np.int8)


def level_codes(rng):
    """Thermometer code: level j+1 = level j with FLIP_FRAC of coords flipped."""
    codes = [rand_bipolar(rng)[0].copy()]
    for _ in range(N_LEVELS - 1):
        nxt = codes[-1].copy()
        flip = rng.choice(D, size=int(FLIP_FRAC * D), replace=False)
        nxt[flip] *= -1
        codes.append(nxt)
    return np.stack(codes)


def bundle(vs):
    s = np.sum(vs.astype(np.int32), axis=0)
    s[s == 0] = 1  # deterministic tie-break
    return np.sign(s).astype(np.int8)


def encode(P, rng):
    """P: [n_words, n_axes] z-scored profiles -> (bound, unbound) hypervectors."""
    axis_ids = rand_bipolar(rng, len(AXES))
    levels = level_codes(rng)
    q = np.digitize(P, BIN_EDGES)  # [n_words, n_axes] in 0..6
    bound, unbound = [], []
    for w in range(P.shape[0]):
        fillers = levels[q[w]]                     # [n_axes, D]
        bound.append(bundle(axis_ids * fillers))   # bind then bundle
        unbound.append(bundle(fillers))            # bundle only
    return np.stack(bound), np.stack(unbound)


def main():
    vecs, layer, rel = load_all()
    P = zscore(project(vecs, AXES))
    profile_upper = sim_upper(P)

    rng = np.random.default_rng(3)
    rhos_b, rhos_u = [], []
    for _ in range(5):  # 5 random codebooks; report mean +/- std
        B, U = encode(P, rng)
        rhos_b.append(spearmanr(profile_upper, sim_upper(B.astype(np.float32))).statistic)
        rhos_u.append(spearmanr(profile_upper, sim_upper(U.astype(np.float32))).statistic)
    rb, ru = float(np.mean(rhos_b)), float(np.mean(rhos_u))
    print(f"layer {layer}, D={D}, {N_LEVELS} levels, 5 codebooks")
    print(f"BOUND   (bind axis x level, then bundle): rho={rb:.4f} +/- {np.std(rhos_b):.4f}")
    print(f"UNBOUND (bundle levels only):             rho={ru:.4f} +/- {np.std(rhos_u):.4f}")
    ok = rb >= 0.6 and rb >= ru + 0.15
    print(f"H5 {'PASS' if ok else 'FAIL'} (need bound>=0.6 and bound>=unbound+0.15)")
    return ok


if __name__ == "__main__":
    main()
