"""falsify.py — version_10 verdict runner.

Pre-registered gates (all fixed before the bank was calibrated):

  T0  BANK      >= 256 axes certified (sign agreement >= 0.95 among committed
                words at coverage >= 10%), AND both v08 meter axes certified.
  T1  FIDELITY  ternary rho > binary rho at K=256, paired on the same axes.
  T2  COMPOSE   >= 7/15 probes hit top-20, p < 0.05 (same bar v09 missed by one).
  T3  LATTICE   exact ternary recovery >= 0.95; unbind-only meter query
                precision AND recall >= 0.95.

Verdict:
  T0 fail          -> DEAD ZONE UNCERTIFIABLE (abstention doesn't stabilize)
  T1 fail          -> ABSTENTION COSTS GEOMETRY (keep binary codes, v09 stands)
  T2 fail          -> STILL NOT COMPOSITIONAL
  all pass         -> TERNARY BARCODE ALIVE — ship as the glyph feature channel
"""
import step1_bank
import step2_fidelity
import step3_compose
import step4_lattice


def main():
    print("=" * 74)
    t0 = step1_bank.main()
    print("=" * 74)
    if not t0:
        print("VERDICT: DEAD ZONE UNCERTIFIABLE — abstention does not yield a "
              "wide certified bank; v09's binary result stands as the best code.")
        return
    t1, _ = step2_fidelity.main()
    print("=" * 74)
    t2 = step3_compose.main()
    print("=" * 74)
    t3 = step4_lattice.main()
    print("=" * 74)

    if t1 and t2 and t3:
        print("VERDICT: TERNARY BARCODE ALIVE — abstention fixes the bit, the "
              "code, the bundle, and the query. Ship it.")
    else:
        parts = [f"T1 {'PASS' if t1 else 'FAIL'}", f"T2 {'PASS' if t2 else 'FAIL'}",
                 f"T3 {'PASS' if t3 else 'FAIL'}"]
        print(f"VERDICT: PARTIAL — {', '.join(parts)}. "
              "See FINDINGS.md for which claims survive.")


if __name__ == "__main__":
    main()
