"""falsify.py — adversarial checks + verdict. Run AFTER step1.

The pilot can die three ways, checked in order:

  F1  MEASUREMENT DEAD — the vectors don't reliably identify concepts.
      (a) median split-half cosine (even vs odd templates) must be > 0.5
      (b) cross-half nearest-neighbor retrieval: half-A vector's nearest
          neighbor among all half-B vectors must be the same word, top-1
          accuracy > 0.8. If measurement is dead, no verdict on the
          hypothesis is possible — everything downstream is noise.

  F2  FRAME CONFOUND — the basis result is an artifact of the metalinguistic
      mention frames (word-form, not meaning). Re-run the basis test on
      usage-frame vectors only; the prime advantage must hold in direction
      (p < 0.25 at k=8 on the noisier half-sized data).

  F3  THE HYPOTHESIS ITSELF — step2 on the full vectors: primes must beat
      frequency-matched random dictionaries with p < 0.05 at k=8, and the
      direction must be consistent at k=4 and k=16.

Verdict:
  MEASUREMENT DEAD  -> rebuild extraction before claiming anything
  HYPOTHESIS DEAD   -> primes are not a privileged linear basis in this
                       model; the weight-space phase is NOT justified
  ALIVE             -> proceed to phase 2 (LoRA weight-space)
"""
import numpy as np

from step2_basis import load_vectors, run_basis_test


def f1_measurement(path="vectors.npz"):
    z = np.load(path, allow_pickle=True)
    sums, counts = z["sums"], z["counts"]
    words = [str(w) for w in z["words"]]
    splits = [str(s) for s in z["split_names"]]
    means = sums / counts[None, :, None, None]
    ia, ib = splits.index("half_a"), splits.index("half_b")

    _, layer, rel, rel_by_layer = load_vectors(path)
    a = means[:, ia, layer, :] - means[:, ia, layer, :].mean(0)
    b = means[:, ib, layer, :] - means[:, ib, layer, :].mean(0)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)

    sim = a @ b.T
    top1 = float((np.argmax(sim, axis=1) == np.arange(len(words))).mean())
    print(f"F1  layer {layer}: median split-half cos = {rel:.3f}  "
          f"(pass > 0.5) | cross-half top-1 retrieval = {top1:.3f} (pass > 0.8)")
    ok = rel > 0.5 and top1 > 0.8
    print(f"F1  {'PASS' if ok else 'FAIL — MEASUREMENT DEAD'}")
    return ok


def f2_frame_confound(path="vectors.npz"):
    vecs, layer, rel, _ = load_vectors(path, split="usage")
    rng = np.random.default_rng(7)
    print(f"F2  usage-frames-only rerun (layer {layer}):")
    res = run_basis_test(vecs, rng, ks=(8,), verbose=True)
    r = res[8]
    ok = r["prime_mean"] > r["null_mean"] and r["p"] < 0.25
    print(f"F2  {'PASS' if ok else 'FAIL — result does not survive without mention frames'}")
    return ok


def f3_hypothesis(path="vectors.npz"):
    vecs, layer, rel, _ = load_vectors(path)
    rng = np.random.default_rng(7)
    print(f"F3  full basis test (layer {layer}):")
    res = run_basis_test(vecs, rng, verbose=True)
    ok = (res[8]["p"] < 0.05
          and res[4]["prime_mean"] > res[4]["null_mean"]
          and res[16]["prime_mean"] > res[16]["null_mean"])
    print(f"F3  {'PASS' if ok else 'FAIL — HYPOTHESIS DEAD'}")
    return ok, res


def main():
    print("=" * 70)
    m_ok = f1_measurement()
    print("=" * 70)
    if not m_ok:
        print("VERDICT: MEASUREMENT DEAD — no claim about the hypothesis is possible.")
        return
    f2_ok = f2_frame_confound()
    print("=" * 70)
    f3_ok, _ = f3_hypothesis()
    print("=" * 70)
    if f3_ok and f2_ok:
        print("VERDICT: ALIVE — primes beat frequency-matched controls; "
              "phase 2 (weight-space) is justified.")
    elif f3_ok and not f2_ok:
        print("VERDICT: CONFOUNDED — the effect exists but may be word-form, "
              "not meaning. Fix frames before phase 2.")
    else:
        print("VERDICT: HYPOTHESIS DEAD — NSM primes are not a privileged "
              "linear basis in this model's activation space.")


if __name__ == "__main__":
    main()
