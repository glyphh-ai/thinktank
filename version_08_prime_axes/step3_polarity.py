"""step3 — H4: are the axes CALIBRATED? (interpretability test)

For the three axes with pre-registered ground truth (good-bad, big-small,
someone-something), test whether labeled target words rank on the correct
side. Statistic: AUC (Mann-Whitney U between the +1 and -1 groups) — rank
based, immune to any offset in the projections. Sign accuracy reported as a
secondary, plus a sanity null: the same labels scored against random
control-pool axes must sit near AUC 0.5.

Pre-registered pass for H4: every labeled axis MW p < 0.05 AND mean AUC >= 0.8.
"""
import numpy as np
from scipy.stats import mannwhitneyu

from axes import AXES, AXIS_NAMES, POLARITY
from step1_project import load_all, project

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "version_07_prime_basis"))
from concepts import CONTROL_POOL, TARGETS  # noqa: E402


def auc_for(proj_col, labels):
    ti = {w: i for i, w in enumerate(TARGETS)}
    pos = [proj_col[ti[w]] for w, s in labels.items() if s > 0]
    neg = [proj_col[ti[w]] for w, s in labels.items() if s < 0]
    u = mannwhitneyu(pos, neg, alternative="greater")
    auc = u.statistic / (len(pos) * len(neg))
    acc = (np.sum(np.array(pos) > 0) + np.sum(np.array(neg) < 0)) / (len(pos) + len(neg))
    return float(auc), float(u.pvalue), float(acc)


def main():
    vecs, layer, rel = load_all()
    P = project(vecs, AXES)
    print(f"layer {layer}\n{'axis':<20s} {'AUC':>6s} {'p':>9s} {'sign-acc':>9s}  n")

    aucs, ps = [], []
    for name, labels in POLARITY.items():
        col = P[:, AXIS_NAMES.index(name)]
        auc, p, acc = auc_for(col, labels)
        aucs.append(auc); ps.append(p)
        print(f"{name:<20s} {auc:>6.3f} {p:>9.2e} {acc:>9.2f}  {len(labels)}")

    # sanity: same labels against random control-pool axes -> should be ~0.5
    rng = np.random.default_rng(7)
    rand_aucs = []
    for _ in range(200):
        idx = rng.choice(len(CONTROL_POOL), size=2, replace=False)
        col = project(vecs, [(CONTROL_POOL[idx[0]], CONTROL_POOL[idx[1]])])[:, 0]
        for labels in POLARITY.values():
            rand_aucs.append(auc_for(col, labels)[0])
    print(f"\nrandom-axis sanity: AUC = {np.mean(rand_aucs):.3f} +/- {np.std(rand_aucs):.3f} (want ~0.5)")

    ok = all(p < 0.05 for p in ps) and np.mean(aucs) >= 0.8
    print(f"H4 {'PASS' if ok else 'FAIL'} (need every p<0.05 and mean AUC>=0.8; "
          f"mean AUC={np.mean(aucs):.3f})")
    return ok


if __name__ == "__main__":
    main()
