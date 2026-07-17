# FINDINGS — version_12_wide_seed

242 named contrast pairs / 458 seed words at layer 12; v10 protocol
verbatim; frozen transfer to the untouched v11 benchmark vectors. Design
committed before results (`git log`). Note: the certified bank is 242 axes,
so "K=256" gates were graded on a 242-bit code against 256-bit SimHash — a
slight handicap we inherit rather than adjust post hoc.

## Verdict: PARTIAL (G1, G4, G5 PASS; G2, G3 FAIL) — and the failure line is a discovery: the barcode is a SIMILARITY instrument, not a relatedness one

| gate | v11 | v12 |
|---|---|---|
| G1 substrate | PASS | **PASS** (layer 12: SimLex 0.4893, WS-353 0.6832) |
| G2 retention ≥90% | 70.3% / 72.6% | **101.5%** on SimLex / 71.0% on WS-353 → FAIL on the pair |
| G3 named ≥ SimHash | FAIL both | **PASS on SimLex** (0.4965 vs 0.4323) / FAIL on WS-353 → FAIL on the pair |
| G4 zero-shot valence ≥0.50 | 0.4901 | **0.5662 PASS** (n=2,925) |
| G5 ridge features | 69.0%, < SimHash | **90.3% PASS**, > SimHash (0.7447 vs 0.6961) |

## The headline: on strict similarity, the named code beats its own substrate

On SimLex-999 — the benchmark built to measure *similarity* and punish mere
association — the 242-bit ternary code scores **0.4965 vs the full 896-dim
embedding's 0.4893** (101.5% retention) and crushes SimHash-242 (0.4440 on
the clean split). With all seed-overlapping pairs removed it holds 84.5%
retention and still beats SimHash (0.4575 vs 0.4440). A 242-bit code where
every bit has a name, out-scoring the embedding it was distilled from, on
the hardest external similarity test, with published word2vec at ~0.44.

The mechanism is visible in the failures: WS-353 (71.0%) and MEN (58.9%)
are *relatedness* benchmarks — coffee~cup, movie~popcorn. Contrast-pair
bits encode shared **attributes** (hot/cold, alive/dead, big/small);
attribute overlap IS similarity, and it systematically strips association.
The code-vs-full agreement is uniform across benchmarks (ρ ≈ 0.71–0.79),
so the divergence from the embedding is the same everywhere — it just
*helps* where humans rated similarity and *hurts* where they rated
relatedness. The barcode isn't a compressed embedding; it is an
**attribute geometry** — a denoised, similarity-specialized view.

## v11's rank diagnosis: confirmed where it applied

Widening seeds 61→458 moved every gate it could move: SimLex retention
70%→101% (84.5% clean), G5 features 69%→90.3% (now beating SimHash), G4
0.490→0.566. What it could not move is *kind*: no number of antonym pairs
encodes co-occurrence, because relatedness is not an attribute contrast.
The residual deficit is structural to the axis type, not to rank.

## Where this leaves the program

Per the pre-registered letter: PARTIAL — G2/G3 required both benchmarks and
relatedness failed them. Per the substance: the strongest possible version
of the auditable-code claim survived external testing, on the benchmark
that most closely matches what the product story needs. Auditable *matching*
("these two records agree on 213 named attributes") is attribute similarity;
relatedness is what unaudited vector search already does well. The honest
positioning writes itself:

> **242 named bits ≈ full-embedding quality for similarity (SimLex 0.497
> vs 0.489), 90% of its downstream feature power, zero-shot human-valence
> at ρ=0.57 — auditable bit-by-bit. Not a relatedness index.**

v13 candidates, in order of leverage:
1. **Ship-shape the similarity instrument** — context (senses), per-token
   codes, the glyph channel; position relatedness out of scope by design.
2. **A second axis type for relatedness** (topic/frame anchors rather than
   contrasts) if the full geometry must be covered — separate channel,
   separate names, same certification discipline.

## Reproduce

```
python pairs.py && python step1_seed_extract.py   # ~8 min CPU
python falsify.py                                 # the gate table above
```
