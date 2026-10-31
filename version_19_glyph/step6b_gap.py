"""step6b_gap.py — count with a per-cue cut instead of one global threshold.

step6's canvas keeps residents separable from non-residents at 16k facts
(98%), but one global threshold drifts as the canvas grows. Here each cue
gets its own cut: sort its similarity scores and cut at the largest gap
among the top K. No retraining; loads step6's saved canvas.

Run:  .venv/bin/python step6b_gap.py
"""

import os

import numpy as np

import step4b_attractor as att
import step6_count as s6

att.P = 8
TOP = 64                               # residents never exceed this here


TAU, BAND = 0.68, 0.15                 # step6's global threshold, and a band around it


def gap_count(sims):
    top = -np.sort(-sims, axis=1)[:, :TOP + 1]
    return np.argmax(top[:, :-1] - top[:, 1:], axis=1) + 1


def hybrid_count(sims):
    """Largest gap whose midpoint sits within BAND of the global threshold."""
    top = -np.sort(-sims, axis=1)[:, :TOP + 1]
    gap = top[:, :-1] - top[:, 1:]
    mid = (top[:, :-1] + top[:, 1:]) / 2
    gap = np.where(abs(mid - TAU) <= BAND, gap, -1)
    return np.argmax(gap, axis=1) + 1


def main():
    m = att.Canvas()
    m.load_weights(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "step6_canvas.safetensors"))
    print(f"{'facts':>7}{'global tau':>12}{'largest gap':>13}{'gap near tau':>14}{'separable':>11}")
    rng = np.random.default_rng(7)
    for n in (64, 256, 1024, 4096, 16384):
        res = [s6.probe(m, rng, n) for _ in range(3)]
        tru = np.concatenate([r[3] for r in res])
        acc = [np.mean(np.concatenate([f(r[2]) for r in res]) == tru)
               for f in (lambda s: (s > TAU).sum(1), gap_count, hybrid_count)]
        print(f"{n:>7,}{acc[0]:>12.3f}{acc[1]:>13.3f}{acc[2]:>14.3f}"
              f"{np.mean([r[4] for r in res]):>11.3f}", flush=True)


if __name__ == "__main__":
    main()
