# Named Bits: an auditable semantic barcode from a frozen tiny language model

*Glyphh AI — thinktank versions 07–12. All numbers below are reproducible
from this repository; every experiment's gates were committed to git before
its results existed (see the `— pre-registered design (no results yet)`
commits in the history).*

## Abstract

We ask whether the ~65 semantic primes of Natural Semantic Metalanguage
(NSM) form a privileged basis for meaning inside a language model — and
answer no, twice, with pre-registered kills. What survives the failures is
more useful than the hypothesis: **a certified bank of named contrast axes**
("meters") read from a frozen Qwen2.5-0.5B on CPU, quantized into **ternary
codes whose every bit has a human-readable name and the right to abstain**.
On external human ground truth, with the entire pipeline frozen on a
disjoint vocabulary: the 242-bit named code **matches the full 896-dim
embedding it was distilled from on SimLex-999** (ρ = 0.497 vs 0.489;
word2vec ≈ 0.44), carries 90.3% of its downstream predictive power for
human valence, and reads valence zero-shot at ρ = 0.566 across 2,925 words
— one dot product, no training. The code is a **similarity instrument, not
a relatedness index** (WS-353/MEN retention ~71%/59%): contrast bits encode
shared attributes, which is what similarity is and association is not.
Codes bundle compositionally (`baby+dog → puppy` at rank 1 of 1025) and
round-trip a hyperdimensional bound memory losslessly at D = 16k. The
result is an end-to-end, bit-auditable semantic feature channel that costs
a frozen 0.5B model and a lookup table.

## 1. The question

The project began as a weight-space idea: fine-tune a tiny model once per
concept, diff the weights against the base, and do algebra on the diffs
with NSM primes as the atomic inventory. Task-vector arithmetic makes the
algebra plausible; the untested part was the basis claim. We gated the
expensive version (~660 fine-tunes) on a cheap activation-space pilot —
which killed it, and every subsequent version was built from the previous
one's autopsy.

## 2. Method spine (constant across versions)

- **Substrate**: frozen `Qwen/Qwen2.5-0.5B`, CPU, forward passes only.
- **Extraction**: a word's vector = mean residual-stream state at its last
  subword token across 12 fixed carrier frames (6 usage, 6 mention), per
  layer; split-half (even/odd frames) kept for reliability and
  certification.
- **Discipline**: every version pre-registers falsifiable gates and commits
  them before results; failures get root-cause autopsies; exploratory
  analyses are labeled and never override pre-registered verdicts; nulls
  are matched (frequency-matched dictionaries, equal-K SimHash, shuffled
  pairings, seed-overlap-free splits).

## 3. What happened, version by version

| v | claim tested | verdict | the number |
|---|---|---|---|
| 07 | primes are dictionary **atoms** | **DEAD** | primes R²=0.218 vs matched-null 0.210±0.006 (p=.075); *no* 61-word dictionary spans >~1/5 of a word vector |
| 07 | primes carry category structure | alive | within/between cos 0.151/0.010, p<10⁻⁴ |
| 08 | primes are coordinate **axes** | **DEAD, inverted** | prime axes ρ=0.334 < random pairs 0.518 — opposites share cos 0.414, their diffs span 6.3% of variance |
| 08 | primes are calibrated **meters** | alive | good–bad AUC 0.964, someone–something 0.920 (pre-registered labels) |
| 08 | binding ≫ bundling as carrier | alive | profile geometry ρ: bound 0.618 vs unbound 0.093 |
| 09 | K named bits ≈ SimHash (fidelity tax) | alive ≤K≈256 | tax −58% at K=16, −60% at 32, +4.7% at 256, +19% at 512 |
| 09 | binary bits compose / carry meters | **DEAD** | 6/15 probes (bar 7); good–bad sign-stability 0.875 — *a meter is not a bit* |
| 10 | ternary bits (certified abstention) | **3/4 PASS** | all 1830 axes certify; ternary>binary at every K; **8/15 probes** (`baby+dog→puppy` #1); lattice recall failure = capacity constant (D=16k → P/R 1.000) |
| 11 | frozen transfer to human benchmarks | **PARTIAL** | substrate legit (SimLex 0.498 > word2vec); barcode capped at ~70% retention — **rank ≤ 60 ceiling** (all axes from 61 seed words); meter real but frozen at wrong depth (ρ 0.490 @L5 vs 0.567 @L12) |
| 12 | wide seeds (242 pairs / 458 words) | **PARTIAL, decisive** | **SimLex: code 0.4965 > substrate 0.4893** (SimHash 0.432); features 90.3% of full, > SimHash; valence 0.566 PASS; relatedness (WS/MEN) structurally out of scope |

Three findings deserve their sentences:

1. **Meaning in this model is high-rank.** No small named inventory spans
   it — not as atoms (v07), not as axes (v08). The compression dream died
   twice with matched nulls. What a named inventory *can* do is measure:
   individual contrast directions are superb instruments at their own cut.
2. **Reliability is not validity.** Split-half reliability peaks at layer 5
   (form-adjacent); agreement with human semantics peaks mid-stack (layer
   12). Instruments must be certified against external ground truth at
   depth, per axis.
3. **A meter is not a bit — unless it can abstain.** Sign-quantizing forces
   the neutral mass to coin-flip. Certified dead zones ({−1, 0, +1}) fixed
   the bank, the fidelity, the composition, and the query in one move.

## 4. The artifact

The surviving pipeline, every link tested:

```
frozen 0.5B LM
  → 12-frame extraction (layer 12)
  → 242 named contrast axes (hot–cold, win–lose, war–peace, …)
  → frozen (μ, σ, τ) tables → ternary code, bits abstain when neutral
  → bundling retrieval (attribute intersection → lexicalized concepts)
  → bipolar bound records (D=16k), threshold-unbind reads,
    feature queries by named bit
```

Supported claims: full-substrate similarity quality at 242 auditable bits
(SimLex 0.497 vs 0.489; 84.5% retention with all seed-overlapping pairs
removed, still > SimHash); 90.3% of the embedding's ridge-predictive power
for human valence; zero-shot valence ρ = 0.566; compositional retrieval at
rank 1; lossless lattice carriage at D = 16k.

## 5. Honest limits and non-claims

- **Not a relatedness index.** Attribute bits strip association by
  construction; WS-353/MEN retention is ~71%/59%. Auditable *matching* is
  attribute similarity — that is the intended niche, not a limitation we
  hide.
- **Not SOTA similarity.** Supervised antonym-injection methods reach
  ~0.74 on SimLex. Our claim is efficiency + auditability at zero training,
  not the leaderboard.
- **Isolated words, English, one model.** Context (word senses), other
  languages, and other substrates are untested (the extraction protocol is
  model-agnostic; nothing else in the pipeline is model-specific).
- **G4/layer provenance**: layer 12 was identified in an exploratory sweep
  that used Warriner data; the valence gate is therefore not fully
  independent of the layer choice (the similarity gates are — layer 12
  slightly *hurts* them vs layer 5).

## 6. Relation to prior work

Task vectors (Ilharco et al.) motivated the origin; the pilot's negative
result is why no fine-tune farm was built. POLAR (Mathew et al.) also
projects embeddings onto antonym axes — we add certified abstention,
frozen-transfer evaluation, the fidelity-tax curve against SimHash (which
random-hyperplane LSH theory says we should lose; we don't below K≈256, and
on SimLex at all), and the VSA/HDC integration with measured capacity.
Binder et al.'s 65 experiential features and counter-fitting's SimLex
results are adjacent anchors. The bind/bundle memory follows the VSA
tradition (Kanerva; this repo's `version_06_hdc_forward`).

## 7. Reproducibility

Each version directory carries `README.md` (design, pre-registered gates),
`falsify.py` (one command → gate table + verdict), and `FINDINGS.md`
(results with autopsies). Extractions regenerate from scratch on CPU
(~10–70 min each). The git history is the audit trail: for v09–v12 the
design commit precedes the results commit.

```
v07 python falsify.py     # atoms die; structure survives
v08 python falsify.py     # axes die; meters + binding survive
v09 python falsify.py     # binary barcode: faithful, not compositional
v10 python falsify.py     # ternary: 3/4, composition alive
v11 python falsify.py     # frozen transfer: rank ceiling found
v12 python falsify.py     # wide seeds: SimLex 0.497 > substrate 0.489
```
