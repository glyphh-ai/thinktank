"""version_10 shared core: the ternary barcode {-1, 0, +1} with a certified
dead zone.

v09's autopsy: a meter is not automatically a bit. Binary codes force every
word to pick a side on every axis; the neutral mass coin-flips (good-bad:
AUC 0.964 at the extremes, sign-stability 0.875 overall). Fix: per axis i,
calibrate a dead-zone width tau_i on split-half data — commit a bit only when
|z| >= tau_i, output 0 otherwise.

Calibration rule (pre-registered, measurement-only — no downstream test
signal): tau_i is the SMALLEST value on a fixed z-quantile grid such that,
among committed words (|z_pooled| >= tau_i), the half-A and half-B projection
signs agree on >= 95%. An axis enters the bank only if some tau_i achieves
that with coverage >= 10% of the vocabulary. Ordering of the bank is the same
purely geometric greedy decorrelation as v09.

A new word's code needs only its own vector: c_i = sign(z_i) if |z_i| >= tau_i
else 0, with the (mean, std, tau) calibration frozen from this vocabulary.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V09 = os.path.join(HERE, "..", "version_09_semantic_barcode")
sys.path.insert(0, V09)

from barcode import (PRIMES, all_prime_pairs, best_layer, eval_vectors,  # noqa: E402,F401
                     pair_axes, word_vectors_v07, cos_sim_upper)
from vocab import PROBES  # noqa: E402,F401

BANK_NPZ = os.path.join(HERE, "ternary_bank.npz")

AGREE_MIN = 0.95
COVERAGE_MIN = 0.10
TAU_GRID = np.round(np.arange(0.0, 2.01, 0.05), 2)


def calibrate(layer):
    """Per prime-pair axis: smallest tau meeting AGREE_MIN with COVERAGE_MIN.

    Returns axes, names, tau, coverage, mu, sd for the surviving bank,
    ordered by greedy geometric decorrelation.
    """
    P = word_vectors_v07(layer, PRIMES)
    pairs = all_prime_pairs()
    A = pair_axes(P, pairs)
    names = [f"{PRIMES[i]}-{PRIMES[j]}" for i, j in pairs]

    _, V = eval_vectors(layer)
    _, Va = eval_vectors(layer, split="half_a")
    _, Vb = eval_vectors(layer, split="half_b")
    proj = V @ A.T
    mu, sd = proj.mean(0), proj.std(0) + 1e-9
    z = (proj - mu) / sd
    agree_ab = np.sign(Va @ A.T) == np.sign(Vb @ A.T)

    keep, taus, covs = [], [], []
    n = len(V)
    for i in range(len(pairs)):
        for tau in TAU_GRID:
            committed = np.abs(z[:, i]) >= tau
            cov = committed.mean()
            if cov < COVERAGE_MIN:
                break
            if agree_ab[committed, i].mean() >= AGREE_MIN:
                keep.append(i); taus.append(tau); covs.append(cov)
                break

    keep = np.array(keep)
    Ak = A[keep]
    order = [0]
    mc = np.abs(Ak @ Ak[0])
    for _ in range(len(keep) - 1):
        mc[order] = np.inf
        nxt = int(np.argmin(mc))
        order.append(nxt)
        mc = np.maximum(mc, np.abs(Ak @ Ak[nxt]))
    o = np.array(order)

    return dict(axes=Ak[o], names=np.array([names[i] for i in keep[o]]),
                tau=np.array(taus)[o], coverage=np.array(covs)[o],
                mu=mu[keep[o]], sd=sd[keep[o]], layer=layer)


def save_bank(bank):
    np.savez_compressed(BANK_NPZ, **bank)


def load_bank():
    z = np.load(BANK_NPZ, allow_pickle=True)
    return {k: z[k] for k in z.files}


def ternary_codes(V, bank, K=None):
    """{-1,0,+1} codes for rows of V on the first K bank axes."""
    K = K or len(bank["axes"])
    z = (V @ bank["axes"][:K].T - bank["mu"][:K]) / bank["sd"][:K]
    c = np.sign(z) * (np.abs(z) >= bank["tau"][:K])
    return c.astype(np.int8)


def binary_codes(V, bank, K=None):
    """Plain sign codes on the same axes (the v09 baseline, paired)."""
    K = K or len(bank["axes"])
    z = (V @ bank["axes"][:K].T - bank["mu"][:K]) / bank["sd"][:K]
    s = np.sign(z)
    s[s == 0] = 1
    return s.astype(np.int8)


def code_cos_upper(C):
    Cf = C.astype(np.float32)
    n = np.linalg.norm(Cf, axis=1, keepdims=True) + 1e-9
    S = (Cf / n) @ (Cf / n).T
    return S[np.triu_indices(len(C), k=1)]
