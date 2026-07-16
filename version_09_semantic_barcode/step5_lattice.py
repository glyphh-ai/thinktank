"""step5 — the lattice holds the barcode: bind 256 named bits, read them back.

Each word's K=256 named barcode is stored as ONE D=8192 bipolar record:

    record(w) = sign( sum_i  bit_i(w) * axis_id_i )      (bind = multiply)

i.e. 256 bound (axis_id, polarity) pairs bundled together — the H5 encoding
at real width. Reading a bit is unbinding: sign(record . axis_id_i).

Tests (pre-registered K4):
  (a) bit recovery — fraction of the 1027 x 256 stored bits read back
      correctly. Need >= 0.99. (SNR argument says ~Phi(sqrt(D/K)) ~ 1.)
  (b) feature query — "valence = negative AND animacy = positive" answered
      ONLY by unbinding records, compared to the ground-truth barcode bits.
      Need precision AND recall >= 0.95. The two query bits are the axes
      v08 certified as meters: good-bad and someone-something (they must be
      in the stable bank for the query to be defined; assert).
"""
import numpy as np

from barcode import codes, eval_vectors, load_axis_bank

K = 256
D = 8192


def main():
    bank, names, stability, layer = load_axis_bank()
    words, V = eval_vectors(layer)
    C = codes(V, bank[:K])  # ground-truth bits
    kept = names[:K]

    rng = np.random.default_rng(5)
    axis_ids = (rng.integers(0, 2, size=(K, D)) * 2 - 1).astype(np.int8)

    # store: one record per word
    records = np.sign(C.astype(np.int32) @ axis_ids.astype(np.int32))
    records[records == 0] = 1

    # (a) read every bit back by unbinding
    read = np.sign(records @ axis_ids.T.astype(np.int32))
    read[read == 0] = 1
    bit_acc = float((read == C).mean())
    print(f"layer {layer}: {len(words)} records, K={K} bits in D={D}")
    print(f"(a) bit recovery: {bit_acc:.4f} (need >= 0.99)")

    # (b) feature query via unbind only
    if "good-bad" not in kept or "someone-something" not in kept:
        print("(b) feature query: the v08 meter axes (good-bad, someone-something) "
              "did not survive stability certification at this layer — query "
              "undefined as pre-registered.")
        print("K4 FAIL (query bits unavailable)")
        return False
    vi, ai = kept.index("good-bad"), kept.index("someone-something")
    truth = (C[:, vi] == -1) & (C[:, ai] == +1)          # negative-valence animates
    pred = (read[:, vi] == -1) & (read[:, ai] == +1)
    tp = int((pred & truth).sum())
    prec = tp / max(int(pred.sum()), 1)
    rec = tp / max(int(truth.sum()), 1)
    print(f"(b) query [valence- AND animate+]: {int(truth.sum())} true members; "
          f"precision {prec:.3f} recall {rec:.3f} (need >= 0.95)")
    sample = [words[i] for i in np.where(pred)[0][:12]]
    print(f"    sample members: {', '.join(sample)}")

    ok = bit_acc >= 0.99 and prec >= 0.95 and rec >= 0.95
    print(f"K4 {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    main()
