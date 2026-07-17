"""falsify.py — version_09 verdict runner.

Pre-registered criteria, fixed before any barcode was computed:

  K1  WIDTH     >= 256 prime-pair axes with split-half sign stability >= 0.90.
                Fewer -> the barcode dies at width; no verdict downstream.
  K2  FIDELITY  at K=256, named-bit Spearman >= 0.80 x SimHash Spearman
                (fidelity tax <= 20% at the working width).
  K3  COMPOSE   >= 7/15 pre-registered bundle probes hit top-20 of ~1025,
                empirical p < 0.05 vs random-input null.
  K4  LATTICE   bit recovery >= 0.99 from bound D=8192 records; unbind-only
                feature query precision AND recall >= 0.95.

Verdict:
  K1 fail             -> BARCODE DEAD (not enough stable named bits)
  K2 fail             -> TAX TOO HIGH (interpretability isn't free here;
                         named bits cost too much geometry at K=256)
  K2 pass, K3 fail    -> FAITHFUL BUT NOT COMPOSITIONAL (usable as codes,
                         bundling doesn't retrieve concepts)
  K2+K3+K4 pass       -> SEMANTIC BARCODE ALIVE
"""
import step2_mine
import step3_fidelity
import step4_compose
import step5_lattice


def main():
    print("=" * 74)
    k1 = step2_mine.main()
    print("=" * 74)
    if not k1:
        print("VERDICT: BARCODE DEAD — fewer than 256 stable named axes.")
        return
    k2, _ = step3_fidelity.main()
    print("=" * 74)
    k3 = step4_compose.main()
    print("=" * 74)
    k4 = step5_lattice.main()
    print("=" * 74)

    if not k2:
        print("VERDICT: TAX TOO HIGH — named bits lose too much geometry vs "
              "SimHash at K=256. Interpretability is not free in this model.")
    elif not k3:
        print("VERDICT: FAITHFUL BUT NOT COMPOSITIONAL — codes carry geometry "
              "cheaply but bundles do not retrieve lexicalized concepts."
              + ("" if k4 else " (K4 also failed — check lattice params.)"))
    elif not k4:
        print("VERDICT: SEMANTICS ALIVE, CARRIER BROKEN — fix D/K before "
              "shipping to the lattice.")
    else:
        print("VERDICT: SEMANTIC BARCODE ALIVE — named bits are cheap enough, "
              "bundles retrieve concepts, and the lattice stores and queries "
              "the codes by unbinding. Ship it as a glyph feature channel.")


if __name__ == "__main__":
    main()
