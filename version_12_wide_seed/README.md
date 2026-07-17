# version_12_wide_seed — lifting the rank ceiling

v11's exact diagnosis: the v10 bank's 1830 axes were all differences of the
same **61** prime vectors — rank ≤ 60 — leaving ~30% of human-relevant
geometry invisible. v12 attacks only that variable:

- **242 hand-curated named contrast pairs** over **458 distinct seed words**
  (`pairs.py`) → rank ceiling ~457. The 13 NSM oppositions are folded in
  under their old names; every new bit stays self-naming (hot-cold,
  win-lose, war-peace...). Pair selection used no benchmark ratings and no
  model signal.
- **Layer 12** (semantic depth), per v11's meter lesson. Disclosure: that
  sweep used Warriner data, so G4 is not fully independent; the similarity
  gates are (layer-12 full-vector SimLex is *lower* than layer 5's, so the
  choice does not favor them).
- Everything else is the v10 protocol verbatim: dead-zone certification on
  the v09 vocabulary (agreement ≥ 0.95, coverage ≥ 0.10), geometric ordering,
  frozen (μ, σ, τ, g09) applied to the untouched v11 benchmark vectors.

## Gates — identical bars to v11

G1 substrate (SimLex ≥ 0.35, WS-353 ≥ 0.55) · G2 retention ≥ 90% at K=256 ·
G3 named ≥ SimHash-256 · G4 zero-shot valence ≥ 0.50 · G5 ridge ≥ 90% of
full and > SimHash. Ungated extras: K=512 retention, and a no-overlap
robustness split (benchmark pairs containing any seed word excluded).

## Run

```
python pairs.py                # inventory sanity
python step1_seed_extract.py   # 286 uncovered seeds, ~15 min CPU
python falsify.py              # builds bank + grades G1-G5
```

Results: see [`FINDINGS.md`](FINDINGS.md).
