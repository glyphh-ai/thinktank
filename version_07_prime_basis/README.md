# version_07_prime_basis — are the NSM primes a privileged basis for meaning in a tiny LM?

Born from a weight-space idea: *fine-tune the tiniest Qwen once per concept on
an identical corpus template, diff the weights against the base, and do algebra
on the diffs — with the NSM semantic primes as the atomic set.* The algebra
exists (task vectors — Ilharco et al. 2022: fine-tunes from one init stay in
one loss basin, so deltas add/subtract meaningfully). The genuinely novel part
is the basis claim: **that the ~65 Natural Semantic Metalanguage primes span
concept space**, i.e. every word's delta decomposes into prime deltas.

That claim is expensive to test in weight space (~400 fine-tunes) and cheap to
test in activation space (~3.5k forward passes, CPU, minutes). If the primes
are not a privileged basis for *representations*, they will not be one for
*weight deltas* — so the activation pilot is the gate. This directory is that
gate.

## Hypothesis

**H1 (privileged basis):** sparse linear combinations of the 61 prime concept
vectors reconstruct arbitrary content-word vectors better than sparse
combinations of 61 frequency-matched random common words.

**H2 (semantic structure):** prime vectors carry NSM's own category structure
(mental predicates cluster with each other, space primes with each other, ...).

Known theoretical ceiling, stated up front: linear combination captures
**bundling** only. NSM explications are structured sentences — they *bind*
(who does what to whom, because of what). There is no binding operation in
either activation arithmetic or weight arithmetic, so even a full win here
validates the bag-of-primes part of the claim, not the syntax. The residual
that sparse coding *cannot* explain is where the binding structure lives —
and that is the bridge back to `version_06_hdc_forward`, where bind/bundle
both exist.

## Method

Model: `Qwen/Qwen2.5-0.5B` (the tiniest Qwen; 24 layers, d=896), stock, CPU,
forward passes only.

| file | what it is |
|---|---|
| `concepts.py` | the inventory: **61** single-word English exponents of the NSM primes (compounds like DON'T WANT folded, one exponent per allolexic doublet), **130**-word frequency-matched control pool, **100** mid-frequency target words, all pairwise disjoint (asserted). 12 carrier templates: 6 *usage* frames + 6 *mention* frames. |
| `step1_vectors.py` | one forward pass per (word, template); read the residual stream at the word's last subword token, every layer. Store split-half (even/odd templates) and usage/mention means → `vectors.npz`. |
| `step2_basis.py` | **THE test.** Center across the vocabulary (kills the carrier frame), unit-norm. Pick the layer by split-half reliability (measurement criterion, blind to the hypothesis). Reconstruct each target via Orthogonal Matching Pursuit (k = 4/8/16 nonzero coefs) from the prime dictionary vs 20 random 61-word control dictionaries. Statistic: mean R²; empirical p. |
| `step3_structure.py` | H2: within- vs between-category cosine among primes, 10k-permutation null. |
| `falsify.py` | the verdict. F1 measurement floor (split-half cos > 0.5, cross-half top-1 retrieval > 0.8), F2 frame confound (result must survive on usage frames alone), F3 the hypothesis (p < 0.05 at k=8, direction consistent at k=4/16). |

## Kill criteria (pre-registered)

- **MEASUREMENT DEAD** — split-half reliability ≤ 0.5 or cross-half retrieval
  ≤ 0.8: the vectors don't identify concepts; no verdict possible.
- **HYPOTHESIS DEAD** — primes fail to beat frequency-matched random
  dictionaries (p ≥ 0.05 at k=8, or direction flips across k): NSM primes are
  not privileged in this model; **phase 2 is not justified**.
- **CONFOUNDED** — wins overall but not on usage frames alone: we measured
  word-form, not meaning; fix frames before any phase-2 spend.

Why frequency-matched controls: the primes are nearly all top-1000 English
words. Beating rare words would prove only that frequent words have
better-estimated vectors. The null must be *typical common words*, including
function words — an honest, hard null.

## Phase 2 (only if the pilot survives): the original weight-space experiment

- ~65 primes + ~200 words + 65 controls, **two seeds each** (~660 runs), on a
  genuinely tiny model (Pythia-70M class; Qwen2.5-0.5B if compute allows).
- LoRA r=8 on one rigidly identical corpus template per concept; the adapter
  delta **is** the concept vector.
- Noise floor = same-concept cross-seed delta distance; every claim must beat it.
- Same OMP dictionary comparison, plus task-vector analogies
  (apply Δ_good − Δ_bad to the base, evaluate behaviorally).
- The letter-level phase from the original idea is dropped: "everything about
  the letter a" trains metalinguistic trivia, not a semantic unit, and the
  tokenizer smears characters anyway.

## Run

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scikit-learn transformers
python step1_vectors.py        # ~5 min CPU, writes vectors.npz
python step2_basis.py          # H1
python step3_structure.py      # H2
python falsify.py              # verdict
```

Results: see [`FINDINGS.md`](FINDINGS.md).
