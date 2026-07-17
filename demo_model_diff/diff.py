"""Semantic model diff — the flagship demo.

What did instruction tuning do to meaning? Barcode the same 1375-word
vocabulary through Qwen2.5-0.5B (base) and Qwen2.5-0.5B-Instruct, using the
SAME 242 named contrast axes built in each model's own space (axes are
seed-vector differences, so each model is measured in its own calibrated
frame), and diff the readings.

Outputs (printed and written to REPORT.md):
  1. axis stability — Spearman between models per axis across the vocab;
     the most/least stable named dimensions under instruction tuning
  2. most-moved words — largest ||delta z|| across all 242 axes
  3. targeted changelog — top signed movers on axes of obvious interest
     (good-bad, polite-rude, safe-dangerous, formal-casual, obey-rebel, ...)
  4. bit flips — words whose COMMITTED bit (|z| >= 1) changed sign: the
     discrete, named change log

This is the origin idea of the whole line — "the algebra of differences
between models" — executed in the space where meaning proved measurable.
"""
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
for v in ("version_07_prime_basis", "version_12_wide_seed"):
    sys.path.insert(0, os.path.join(HERE, "..", v))
from pairs import ALL_PAIRS, PAIR_NAMES, seed_words  # noqa: E402

LAYER = 12
COMMIT_Z = 1.0
INTEREST = ["good-bad", "polite-rude", "safe-dangerous", "formal-casual",
            "obey-rebel", "true-false", "kind-cruel", "calm-angry",
            "friendly-hostile", "honest-dishonest"]

BASE_SOURCES = [
    os.path.join(HERE, "..", "version_09_semantic_barcode", "vocab_vectors.npz"),
    os.path.join(HERE, "..", "version_12_wide_seed", "seed_vectors.npz"),
    os.path.join(HERE, "..", "version_07_prime_basis", "vectors.npz"),
]
INSTRUCT_NPZ = os.path.join(HERE, "instruct_vectors.npz")
WORDS = [w.strip() for w in open(os.path.join(HERE, "diff_wordlist.txt")) if w.strip()]


def pooled_lut(npz):
    z = np.load(npz, allow_pickle=True)
    words = [str(w) for w in z["words"]]
    splits = [str(s) for s in z["split_names"]]
    m = z["sums"] / z["counts"][None, :, None, None]
    ia, ib = splits.index("half_a"), splits.index("half_b")
    P = (m[:, ia, LAYER, :] + m[:, ib, LAYER, :]) / 2
    return {w: P[i] for i, w in enumerate(words)}


def model_readings(lut):
    """z-scored readings [n_words, n_axes] of WORDS on axes built in-model."""
    missing = [w for w in WORDS if w not in lut]
    assert not missing, f"missing vectors: {missing[:10]}"
    A = np.stack([lut[a] - lut[b] for a, b in ALL_PAIRS])
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    V = np.stack([lut[w] for w in WORDS])
    V = V - V.mean(0)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    proj = V @ A.T
    return (proj - proj.mean(0)) / (proj.std(0) + 1e-9)


def main():
    base_lut = {}
    for src in BASE_SOURCES:
        for w, v in pooled_lut(src).items():
            base_lut.setdefault(w, v)
    zb = model_readings(base_lut)
    zi = model_readings(pooled_lut(INSTRUCT_NPZ))
    dz = zi - zb

    lines = ["# Semantic diff: Qwen2.5-0.5B → Qwen2.5-0.5B-Instruct",
             "",
             f"{len(WORDS)} words x {len(ALL_PAIRS)} named axes, layer {LAYER}. "
             "Each model measured in its own frame (axes rebuilt from its own "
             "seed vectors); readings z-scored within model; diff = z_inst − z_base.",
             ""]

    stab = np.array([spearmanr(zb[:, i], zi[:, i]).statistic for i in range(len(ALL_PAIRS))])
    order = np.argsort(stab)
    lines += ["## Axis stability under instruction tuning",
              f"median axis correlation base↔instruct: **{np.median(stab):.3f}**", "",
              "most changed axes | most stable axes", "---|---"]
    for k in range(8):
        lines.append(f"{PAIR_NAMES[order[k]]} ({stab[order[k]]:.3f}) | "
                     f"{PAIR_NAMES[order[-1-k]]} ({stab[order[-1-k]]:.3f})")

    move = np.linalg.norm(dz, axis=1)
    top = np.argsort(-move)[:20]
    lines += ["", "## Most semantically moved words (||Δz|| over 242 axes)", ""]
    lines.append(", ".join(f"**{WORDS[i]}** ({move[i]:.1f})" for i in top))

    lines += ["", "## Targeted changelog", ""]
    for name in INTEREST:
        if name not in PAIR_NAMES:
            continue
        i = PAIR_NAMES.index(name)
        d = dz[:, i]
        up = np.argsort(-d)[:6]
        dn = np.argsort(d)[:6]
        a, b = name.split("-", 1)
        lines.append(f"**{name}** (axis corr {stab[i]:.2f}): toward *{a}*: "
                     + ", ".join(f"{WORDS[j]} (+{d[j]:.1f})" for j in up))
        lines.append(f"  toward *{b}*: " + ", ".join(f"{WORDS[j]} ({d[j]:.1f})" for j in dn))
        lines.append("")

    cb = np.sign(zb) * (np.abs(zb) >= COMMIT_Z)
    ci = np.sign(zi) * (np.abs(zi) >= COMMIT_Z)
    flips = (cb * ci) < 0  # committed in both, opposite signs
    n_flips = int(flips.sum())
    lines += ["## Committed bit flips (|z| >= 1 in both models, sign reversed)",
              f"{n_flips} flips across {len(WORDS)}x{len(ALL_PAIRS)} bits "
              f"({n_flips / flips.size:.2%})", ""]
    fw, fa = np.where(flips)
    strength = np.abs(dz[fw, fa])
    for k in np.argsort(-strength)[:15]:
        w, i = fw[k], fa[k]
        a, b = PAIR_NAMES[i].split("-", 1)
        frm, to = (a, b) if cb[w, i] > 0 else (b, a)
        lines.append(f"- **{WORDS[w]}**: {PAIR_NAMES[i]} flipped {frm} → {to} "
                     f"(Δz {dz[w, i]:+.1f})")

    report = "\n".join(lines) + "\n"
    open(os.path.join(HERE, "REPORT.md"), "w").write(report)
    print(report)


if __name__ == "__main__":
    main()
