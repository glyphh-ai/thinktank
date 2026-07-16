# version_08_prime_axes — primes as contrast axes, not dictionary atoms

Sequel to [`version_07_prime_basis`](../version_07_prime_basis), which killed
"a word is a sparse bundle of NSM primes" (no 61-word dictionary spans more
than ~1/5 of a word vector) but found the primes strongly organized as
near-antipodal contrast pairs (above~below cos 0.82, good~bad 0.69). v08
tests the reframe: **the primes act as axes the model measures words
against.** No new forward passes — reuses v07's `vectors.npz` (rerun v07
`step1_vectors.py` if missing).

## Hypotheses (criteria fixed before any projection was computed)

**H3 (axes as coordinates):** projections onto 13 NSM opposition axes
(good−bad, big−small, above−below, ...) preserve the cosine geometry of the
100 target words better than (A) 13 random control-pool pairs and (B) the
same 26 prime words re-paired at random. Pass: p < 0.05 vs both nulls.

**H4 (axes as calibrated meters):** on hand-coded ground truth written before
looking (valence n=26, size n=11, animacy n=20), labeled words rank on the
correct side — every axis Mann-Whitney p < 0.05 and mean AUC ≥ 0.8.

**H5 (the v06 crossover — carrier):** a word's 13-axis profile encoded into
the Glyph HDC lattice as `bundle_i(bind(axis_i, level_i))` preserves profile
geometry (Spearman ρ ≥ 0.6) and beats bundling without binding by ≥ 0.15.

| file | what it is |
|---|---|
| `axes.py` | the 13 prime opposition axes + pre-registered polarity labels (57 (axis, word) labels; ambiguous items deliberately unlabeled). Validated against v07's inventory. |
| `step1_project.py` | axis = unit-norm v(A)−v(B) at v07's layer 18; 100 targets → 13-dim profiles; prints per-axis extremes for eyeballing. |
| `step2_geometry.py` | H3: Spearman ρ between full-space and profile-space similarity, vs null A (random pairs), null B (shuffled prime pairing), and a PCA(13) ceiling. |
| `step3_polarity.py` | H4: AUC per labeled axis (rank-based, offset-immune) + random-axis sanity (~0.5). |
| `step4_hdc.py` | H5: bound vs unbound hypervector encoding, D=8192, 7-level thermometer code, 5 codebooks. HDC ops per `version_06_hdc_forward/hdc.py` conventions. |
| `falsify.py` | runs H3 → H4 → H5, prints the verdict. |

## Run

```
python step1_project.py   # eyeball table
python falsify.py         # verdict
```

Results: see [`FINDINGS.md`](FINDINGS.md). Short version: H3 **inverted** —
prime axes are *worse* coordinates than random pairs, because true opposites
share almost everything (within-pair cos 0.414; the 13 axes span 6.3% of
target variance). H4's best axes are superbly calibrated (valence AUC 0.964,
animacy 0.920). H5 passed decisively (bound 0.62 vs unbound 0.09). Primes are
**meters, not maps** — and binding, not bundling, is what carries them.
