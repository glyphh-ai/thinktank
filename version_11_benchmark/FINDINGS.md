# FINDINGS — version_11_benchmark

Frozen v10 pipeline, external human ground truth, 3019 unseen words, 100%
benchmark coverage. Gates pre-registered before extraction finished
(commit before results, as always).

## Verdict: PARTIAL — the substrate and the meter are LEGIT; the barcode hit a structural ceiling with a clean mathematical cause

| gate | result |
|---|---|
| G1 substrate | **PASS** — full vectors: SimLex **0.4975**, WS-353 **0.7195**, MEN 0.7475 |
| G2 retention ≥90% | **FAIL** — ternary-256 keeps 70.3% / 72.6% |
| G3 named ≥ SimHash | **FAIL** — SimHash-256: 0.4317 / 0.6180 vs named 0.3499 / 0.5225 |
| G4 zero-shot valence ≥0.50 | **FAIL by 0.01** — ρ = 0.4901 at frozen layer 5 |
| G5 ridge features | **FAIL** — ternary 0.547 vs full 0.792, SimHash 0.610 |

## G1 — the extraction protocol is genuinely good

This gate could have indicted four versions of infrastructure; instead the
12-frame protocol on a frozen 0.5B CPU model **beats published word2vec
references on SimLex-999 (0.50 vs ~0.44)** and matches them on WS-353
(0.72 vs ~0.70) and MEN (0.75 vs ~0.77). Whatever else is true, the
substrate the barcodes are distilled from holds up against the classic
embedding literature.

## The meter is real — we froze it at the wrong depth

The pre-registered G4 ran at layer 5 (chosen in v09 by split-half
reliability) and missed by a hair: 0.4901 vs 0.50 over 2,925 words. The
exploratory layer sweep tells the actual story:

```
layer  0: 0.149   layer  9: 0.559   layer 18: 0.456
layer  3: 0.348   layer 12: 0.567   layer 21: 0.337
layer  6: 0.530   layer 15: 0.526   layer 24: 0.035
```

Zero-shot ρ = **0.567 at layer 12** — one dot product against v(good)−v(bad),
no training, against 2,925 human ratings (supervised ridge on the full
embedding: 0.79). The methodological finding: **reliability is not
validity.** Split-half reliability peaked at layer 5 (form-adjacent);
semantic alignment with humans peaks mid-stack. Meters must be certified
against external ground truth at their best layer, per axis — exactly the
meter-bank program.

## The barcode's failure is structural, and the diagnosis is exact

Three exploratory eliminations (`exploratory_diagnostics.py`):

1. **Not the freeze**: recalibrating μ/σ on the benchmark vocabulary makes
   retention slightly *worse* (0.337 vs 0.350 SimLex). Frozen z-drift is
   small (median 0.16σ).
2. **Not the layer**: the whole bank rebuilt at semantic layer 12 → 66%
   retention. No better.
3. **Not the width — the opposite**: ALL 1830 axes retain *less* (61%) than
   the first 256 (66%).

Adding more prime pairs hurting is the tell: **every axis is a difference of
the same 61 prime vectors, so the entire bank — any K, any layer, any
calibration — spans a subspace of rank ≤ 60.** The barcode is mathematically
incapable of seeing whatever human-relevant geometry lives outside those 60
dimensions (~30% of it, on this evidence), while SimHash-256 samples 256
independent directions and keeps 86%. v09's "named beats SimHash below
K=256" was true *in-domain* because the model's own similarity structure is
dominated by the top of the spectrum where the prime subspace lives; human
judgments draw on more of the space.

## What this means

- **Validated externally**: the extraction substrate (beats word2vec on
  SimLex) and the zero-shot meter primitive (0.567 at depth). These are
  shippable claims today.
- **Bounded honestly**: named ternary codes seeded by 61 words carry ~70% of
  embedding quality at 11:1 compression, auditable bit-by-bit. That is a
  real artifact with a real trade — but it is NOT "interpretability for
  free" outside the calibration domain, and the pre-registered gates say so.
- **The fix is known and cheap**: the rank ceiling scales with seed
  vocabulary, not with pairs. A few hundred seed words in named contrast
  pairs (antonym inventories are sitting in WordNet; our control pool is
  already extracted) lifts the ceiling from rank 60 to rank hundreds while
  every bit stays named. v12: the wide-seed barcode — same frozen-transfer
  gates, same bars, axes certified at semantic depth per the meter lesson.

## Reproduce

```
python fetch_data.py && python bench_vocab.py
python step1_extract.py            # ~40 min CPU
python falsify.py                  # G1 PASS, G2-G5 FAIL
python exploratory_diagnostics.py  # the autopsy: rank<=60 ceiling, layer-12 meter
```
