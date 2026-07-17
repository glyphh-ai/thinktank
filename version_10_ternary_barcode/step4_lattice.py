"""step4 — T3: the lattice carries TERNARY codes; the v09 K4 query, rescued.

Storage: one bipolar D=8192 record per word, bundling ONLY committed bits:
    record(w) = sign( sum_{i: bit_i != 0} bit_i * axis_id_i )
Sparser bundles -> better SNR per stored bit.

Read rule (pre-registered): s_i = record . axis_id_i / D. A committed bit in
a bundle of n_w items has expected |s| ~ sqrt(2/(pi*n_w)); an uncommitted
axis has |s| ~ 1/sqrt(D). Threshold theta_w = 0.5*sqrt(2/(pi*n_w)) (n_w
stored per record, like a norm): |s| < theta_w -> read 0, else sign(s).

Gates: (a) exact three-way bit recovery >= 0.95 over all words x K bits;
(b) query "good-bad = -1 AND someone-something = +1" (committed-negative
valence AND committed-animate — with the dead zone this means ACTUALLY
negative, not merely below average) answered by unbinding only, precision
and recall >= 0.95 vs the ground-truth ternary bits.
"""
import numpy as np

from ternary import eval_vectors, load_bank, ternary_codes

K = 256
D = 8192


def main():
    bank = load_bank()
    layer = int(bank["layer"])
    names = list(bank["names"][:K])
    words, V = eval_vectors(layer)
    C = ternary_codes(V, bank, K)

    rng = np.random.default_rng(5)
    axis_ids = (rng.integers(0, 2, size=(K, D)) * 2 - 1).astype(np.int8)

    records = np.sign(C.astype(np.int32) @ axis_ids.astype(np.int32))
    records[records == 0] = 1
    n_w = np.abs(C).sum(1)  # committed-bit count per word (stored metadata)

    s = (records @ axis_ids.T.astype(np.int32)).astype(np.float64) / D
    theta = 0.5 * np.sqrt(2 / (np.pi * np.maximum(n_w, 1)))[:, None]
    read = np.where(np.abs(s) < theta, 0, np.sign(s)).astype(np.int8)

    bit_acc = float((read == C).mean())
    committed_acc = float((read == C)[C != 0].mean())
    zero_acc = float((read == C)[C == 0].mean())
    print(f"layer {layer}: {len(words)} records, K={K} ternary bits in D={D}, "
          f"median committed bits/word {int(np.median(n_w))}")
    print(f"(a) exact ternary recovery: {bit_acc:.4f} "
          f"(committed {committed_acc:.4f}, zeros {zero_acc:.4f}; need >= 0.95)")

    if "good-bad" not in names or "someone-something" not in names:
        print("(b) meter axes not in bank — query undefined")
        print("T3 FAIL")
        return False
    vi, ai = names.index("good-bad"), names.index("someone-something")
    truth = (C[:, vi] == -1) & (C[:, ai] == +1)
    pred = (read[:, vi] == -1) & (read[:, ai] == +1)
    tp = int((pred & truth).sum())
    prec = tp / max(int(pred.sum()), 1)
    rec = tp / max(int(truth.sum()), 1)
    members = [words[i] for i in np.where(pred)[0][:12]]
    print(f"(b) query [valence=-1 AND animate=+1]: {int(truth.sum())} true, "
          f"{int(pred.sum())} retrieved; precision {prec:.3f} recall {rec:.3f}")
    print(f"    members: {', '.join(members)}")

    ok = bit_acc >= 0.95 and prec >= 0.95 and rec >= 0.95
    print(f"T3 {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    main()
