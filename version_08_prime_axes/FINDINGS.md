# FINDINGS — version_08_prime_axes

Same substrate as v07: `Qwen/Qwen2.5-0.5B`, layer 18 (split-half reliability
0.911). All criteria pre-registered in `axes.py` / `falsify.py` before any
projection was computed.

## Verdict: AXES-AS-COORDINATES DEAD (inverted, earned). AXES-AS-METERS ALIVE. LATTICE CARRIER CONFIRMED.

## H3 — axes as a coordinate system: DEAD, and it *inverted*

Spearman ρ between full 896-dim target geometry and the 13-dim projection
view:

| view | ρ | p vs primes |
|---|---|---|
| 13 NSM prime axes | **0.334** | — |
| null A: 13 random control-pool pairs | 0.518 ± 0.043 | **1.000** |
| null B: same 26 primes, shuffled pairing | 0.440 ± 0.025 | **1.000** |
| ceiling: PCA(13) on targets | 0.697 | — |

Not only did the prime axes fail to beat random pairs — **every one of 400
null resamples beat the prime axes.** Shuffling the NSM pairings *improves*
geometry preservation. The autopsy explains it:

- within-pair cosine of the NSM oppositions: **0.414** (random pairs: 0.023).
  True opposites are nearly the same vector — good~bad share evaluation-ness,
  above~below share vertical-ness. The difference v(A)−v(B) cancels
  everything shared and keeps only the thin blade that separates them.
- consequence: the 13 prime axes jointly span **6.3%** of target-vector
  variance (mean inter-axis |corr| only 0.158 — they're thin *and* mostly
  non-overlapping; there just isn't much of them).

So the axes are not a map of meaning space. They are 13 narrow, specialized
cuts through it. Random word pairs make better *coordinates* precisely
because random words share nothing — their differences are fat, generic
directions.

## H4 — axes as calibrated meters: the strict gate FAILED, the phenomenon is REAL

| axis | AUC | MW p | sign-acc | n |
|---|---|---|---|---|
| good−bad | **0.964** | 3.2e-05 | 0.92 | 26 |
| someone−something | **0.920** | 8.5e-04 | 0.75 | 20 |
| big−small | 0.643 | 0.26 | 0.64 | 11 |

Random-axis sanity: AUC 0.504 ± 0.205. Pre-registered gate (every axis
p<0.05) fails on big−small, so H4 is formally FAIL — but valence and animacy
are about as calibrated as activation probes get. A 0.5B model's good−bad
axis ranks love/hope/rich above sad/burn/break at AUC 0.96 from *one prime
pair*, no training. big−small plausibly fails for the same reason H3 died:
"big" and "small" are the most similar descriptor pair, and physical size is
weakly encoded at layer 18 (blood/hot/jump projected "big" — it may be an
intensity axis here, not a size axis).

## H5 — the lattice can carry the profiles: PASS, decisively

Encoding 13-axis profiles into D=8192 bipolar hypervectors:

| encoding | ρ (profile geometry preserved) |
|---|---|
| BOUND — bundleᵢ(bind(axisᵢ, levelᵢ)) | **0.618 ± 0.007** |
| UNBOUND — bundleᵢ(levelᵢ) | 0.093 ± 0.001 |

Binding is not an implementation detail; it is 85% of the signal. A bag of
the same levels without axis identity carries almost nothing. This is the
v06 thesis showing up in v08's data: **structure lives in the binding.**

## The arc, and what it means

- v07: primes are not dictionary **atoms** (nothing is — word vectors are
  high-rank).
- v08: primes are not coordinate **axes** either — NSM opposites are too
  similar to span anything; their differences are thin blades.
- What survives both kills: primes as **meters** — individually calibrated,
  interpretable probes (valence 0.96, animacy 0.92) — and the HDC lattice as
  the only tested encoding that carries structured profiles (bind ≫ bundle).

The compression dream (a small closed basis of meaning inside an LM) is now
dead twice, in two different geometries, with pre-registered criteria both
times. The productive residue is a toolkit: pick the prime axes that are
demonstrably calibrated, use them as cheap semantic meters on any word or
sentence, and store/compose the readings in the lattice where binding
preserves what bundling destroys. That is a v06-native feature, not a new
theory of meaning.

## If there is a version_09 in this line

Not more basis-hunting. Two concrete directions the data supports:
1. **Meter mining** — systematically find *which* prime (or non-prime) pairs
   are calibrated meters at which layers, with held-out human norms
   (valence/arousal/concreteness) as ground truth. v08 shows at least two
   exist; the question is the inventory, not the existence.
2. **Bound profiles as glyph features** — ship the H5 encoding: per-token
   axis profiles, bound into the runtime lattice, queried by unbind — a
   semantic annotation channel alongside v06's factual grounding channel.

## Reproduce

```
cd version_07_prime_basis && python step1_vectors.py   # if vectors.npz missing
cd ../version_08_prime_axes && python falsify.py
```
