# FINDINGS — version_10_ternary_barcode

Same substrate as v09 (frozen Qwen2.5-0.5B, 1027 words, blind layer 5).
Design committed before calibration; all gates pre-registered.

## Verdict: 3 of 4 gates PASS — composition is ALIVE at 8/15, and the one FAIL is a capacity constant, not a design flaw

| gate | v09 (binary) | v10 (ternary) |
|---|---|---|
| bank width | 787/1830 stable | **1830/1830 certified** (T0 PASS; good-bad back in at τ=0.15, coverage 0.86) |
| fidelity | tax 4.7% at K=256 | **ternary > binary at every K, paired** (T1 PASS: 0.5981 vs 0.5861 at 256) |
| composition | 6/15 — FAIL by one | **8/15 — T2 PASS** (null 0.43 ± 0.69, p=0.000) |
| lattice | recovery 1.0, query undefined | recovery 0.9911, query P 0.994 / **R 0.925** — T3 FAIL at D=8192; **P/R 1.000 at D=16384** (exploratory) |

## T0/T1 — abstention certifies everything and costs nothing

The dead zone needed to hit 95% split-half sign agreement is thin (median
τ = 0.10σ, median coverage 91%) — which certifies **all 1830** prime-pair
axes, including both v08 meters that binary stability had rejected. The
diagnosis from v09 was exactly right: the instability lived in a narrow
neutral band, and clipping just that band fixes the bit. Paired on identical
axes, ternary codes beat binary at every K — abstention adds information
(a 0 now *means* neutral) instead of subtracting it.

## T2 — composition, third time's the charm

8/15 pre-registered probes hit top-20 (bar: 7). The abstain-on-conflict
bundle (+1 + −1 → 0, no more coin flips) is the difference — same probes,
same vocabulary, same bar v09 missed:

```
baby+dog    -> puppy    rank 1      war+person   -> soldier  rank 1
sick+place  -> clinic   rank 2      young+woman  -> girl     rank 3
school+person -> student rank 4     royal+man    -> prince   rank 7
cold+season -> winter   rank 9      young+man    -> boy      rank 15
```

Two probes landed their target at **rank 1 of 1025**. Bundling named ternary
bits is a real composition mechanism: the intersection of two codes'
committed features points at the lexicalized concept.

## T3 — honest FAIL at the pre-registered D, solved by one doubling

At D=8192 with a median of 231 committed bits per record, threshold-unbind
reads run out of SNR: recovery 0.9911 (gate 0.95 ✓) but query recall 0.925
(gate 0.95 ✗). The pre-registered verdict is FAIL and stands. The exploratory
D-sweep (`exploratory_D_sweep.py`) shows it is purely capacity:

```
D= 8192: recovery 0.9911, query P 0.994 R 0.925
D=16384: recovery 0.9996, query P 1.000 R 1.000
D=32768: recovery 1.0000, query P 1.000 R 1.000
```

Production constant for the glyph runtime: **D ≥ 16k for K=256
mostly-committed ternary bits** (or keep D=8192 and commit fewer bits — the
thin dead zone means records are nearly dense; a fatter τ would trade
coverage for SNR).

One flag for meaning (not gated here): at layer 5 the set "committed-negative
on good-bad AND committed-animate" includes register artifacts (function-y
words). The carrier round-trips bits perfectly at 16k; whether *layer-5*
bits mean what their names say at production quality is the meter-mining
question — certify per (axis, layer) against human norms before shipping
name-level semantics.

## The arc, five versions in

- v07: primes are not dictionary **atoms** (nothing is — word identity is high-rank).
- v08: primes are not coordinate **axes** (opposites are thin blades) — but
  they are calibrated **meters**, and binding ≫ bundling as a carrier.
- v09: K named meters at sign precision = a **barcode**; interpretability is
  *free below K=256* vs SimHash; a meter is not automatically a bit.
- v10: let bits **abstain** and everything clicks — every axis certifies,
  ternary beats binary bit-for-bit, bundles retrieve concepts (8/15, two at
  rank 1), and the lattice carries it all at D=16k.

What exists now, end to end and each link tested: frozen tiny LM → named
prime-pair axes → certified ternary barcode → abstain-on-conflict bundling →
bound bipolar records → threshold-unbind reads and feature queries. That is
a working, auditable semantic feature channel for the glyphh runtime.

What v11 should be, if the line continues: out of the word-list sandbox —
codes for words *in context* (does the barcode of "bank" split by sense?),
per-token profiles over real sentences, and meter certification against
human norms (Warriner/Brysbaert) per (axis, layer). That is the step from
"validated mechanism" to "production instrument."

## Reproduce

```
python falsify.py              # T0 PASS, T1 PASS, T2 PASS (8/15), T3 FAIL at D=8192
python exploratory_D_sweep.py  # D=16384 -> recovery 0.9996, query P/R 1.000
```
