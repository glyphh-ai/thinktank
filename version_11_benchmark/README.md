# version_11_benchmark — the barcode against the outside world

v07–v10 were sealed experiments: our vocabulary, our probes, our geometry.
v11 is the legitimacy test the line has earned: **frozen pipeline, external
benchmarks, human ground truth, numbers comparable to the published
literature.**

## What is frozen (nothing is fit here)

The entire v10 production path, calibrated on a *disjoint* web-frequency
vocabulary: layer 5, the 1830 named prime-pair axes and their order, per-axis
(μ, σ, τ) dead-zone tables, and the centering mean. A benchmark word — never
seen by any calibration step — gets its barcode from one extraction plus a
lookup table (`frozen.py`).

## External ground truth

| dataset | what humans rated | n |
|---|---|---|
| SimLex-999 | strict semantic similarity | 999 pairs |
| WordSim-353 | similarity/relatedness | 353 pairs |
| MEN | relatedness | 3000 pairs (reported, ungated) |
| Warriner et al. 2013 | valence, 1–9 | ~2.9k of our extracted words |

Extraction vocabulary: all 1819 unique benchmark words + a valence-stratified
Warriner sample = **3019 words** (`bench_wordlist.txt`), 100% coverage — no
dropped pairs. Data fetched by `fetch_data.py` (standard research mirrors),
not committed.

## Pre-registered gates

- **G1 substrate**: full 896-d vectors reach SimLex ρ ≥ 0.35 and WS-353
  ρ ≥ 0.55 (word2vec references: ~0.44 / ~0.70). If the substrate fails,
  nothing downstream can be graded.
- **G2 retention**: the 256-bit ternary barcode keeps ≥ 90% of the
  full-vector ρ on SimLex and WS-353 — 11:1 compression into *named* bits.
- **G3 named ≥ anonymous**: barcode ≥ SimHash-256 (5 seeds) on both.
- **G4 zero-shot meter**: the frozen good-bad z-score correlates with human
  valence at ρ ≥ 0.50 across ~2.9k words — one dot product, no training.
- **G5 features**: out-of-fold ridge on the 256 ternary bits predicts human
  valence at ≥ 90% of the full-vector ridge, and beats SimHash-256.

| file | what it is |
|---|---|
| `fetch_data.py` / `bench_vocab.py` | data + committed word list (selection uses only the datasets) |
| `step1_extract.py` | the 12-frame protocol over 3019 words (~36k passes) |
| `frozen.py` | the frozen v10 pipeline applied to unseen words |
| `step2_similarity.py` | G1–G3 |
| `step3_norms.py` | G4–G5 |
| `falsify.py` | verdict |

## Run

```
python fetch_data.py && python bench_vocab.py
python step1_extract.py      # ~70 min CPU
python falsify.py
```

Results: see [`FINDINGS.md`](FINDINGS.md).
