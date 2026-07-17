"""falsify.py — version_08 verdict runner.

Pre-registered criteria (fixed before any projection was computed):

  H3  GEOMETRY   — 13 prime axes preserve target geometry better than
                   (A) 13 random control-pool pairs AND (B) the same 26
                   prime words re-paired at random. p < 0.05 vs both.
  H4  CALIBRATION— every labeled axis (good-bad, big-small,
                   someone-something) MW p < 0.05 and mean AUC >= 0.8.
  H5  CARRIER    — bound HDC encoding rho >= 0.6 and >= unbound + 0.15.

Verdict:
  H3 fail            -> AXES DEAD: primes-as-axes no better than any words;
                        the v07 reframe was wrong too. Line closed.
  H3 pass, H4 fail   -> UNCALIBRATED: axes structure the space but don't
                        mean what NSM says they mean.
  H3+H4 pass         -> AXES ALIVE. H5 then decides whether the lattice
                        can carry them (v06 crossover go/no-go).
"""
import step2_geometry
import step3_polarity
import step4_hdc


def main():
    print("=" * 70)
    rho, pA, pB = step2_geometry.main()
    h3 = pA < 0.05 and pB < 0.05
    print("=" * 70)
    h4 = step3_polarity.main()
    print("=" * 70)
    h5 = step4_hdc.main()
    print("=" * 70)

    if not h3:
        print("VERDICT: AXES DEAD — prime axes are not privileged views of "
              "meaning; the v07 reframe dies with them.")
    elif not h4:
        print("VERDICT: UNCALIBRATED — axes organize the space but do not "
              "carry their NSM meanings. Investigate before building on them.")
    elif h5:
        print("VERDICT: AXES ALIVE + LATTICE-READY — primes-as-axes holds and "
              "the bound HDC encoding carries the profiles faithfully. "
              "v06 crossover is a go.")
    else:
        print("VERDICT: AXES ALIVE, CARRIER WEAK — semantics holds but the "
              "HDC encoding loses geometry; tune D/levels before crossover.")


if __name__ == "__main__":
    main()
