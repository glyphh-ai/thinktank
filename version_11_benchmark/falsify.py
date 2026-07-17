"""falsify.py — version_11 verdict: the barcode against the outside world.

Everything upstream is FROZEN (v10 bank: layer, axes, order, mu/sigma/tau,
centering mean — calibrated on a disjoint web-frequency vocabulary). The
benchmark words are unseen. Gates, fixed before extraction finished:

  G1 SUBSTRATE  full vectors: SimLex >= 0.35, WS-353 >= 0.55
  G2 RETENTION  ternary-256 keeps >= 90% of full rho on SimLex AND WS-353
  G3 NAMED      ternary-256 >= SimHash-256 on SimLex AND WS-353
  G4 ZERO-SHOT  good-bad z-score vs Warriner valence: rho >= 0.50
  G5 FEATURES   ridge on ternary-256 >= 0.90 x full ridge AND > simhash ridge

Verdict:
  G1 fail -> SUBSTRATE WEAK (the 0.5B protocol itself underperforms; barcode
             claims are moot on this substrate)
  G1 pass, any of G2-G5 fail -> PARTIAL (see which claims survive)
  all pass -> LEGIT — the frozen barcode transfers to external human ground
             truth at production settings.
"""
import step2_similarity
import step3_norms


def main():
    print("=" * 74)
    g1, g2, g3, _ = step2_similarity.main()
    print("=" * 74)
    g4, g5 = step3_norms.main()
    print("=" * 74)

    if not g1:
        print("VERDICT: SUBSTRATE WEAK — full vectors underperform the gate; "
              "no barcode claim can be graded on this substrate.")
    elif all([g2, g3, g4, g5]):
        print("VERDICT: LEGIT — frozen named ternary barcode holds up on "
              "external human benchmarks at production settings.")
    else:
        flags = [f"G2 {'PASS' if g2 else 'FAIL'}", f"G3 {'PASS' if g3 else 'FAIL'}",
                 f"G4 {'PASS' if g4 else 'FAIL'}", f"G5 {'PASS' if g5 else 'FAIL'}"]
        print(f"VERDICT: PARTIAL — {', '.join(flags)}.")


if __name__ == "__main__":
    main()
