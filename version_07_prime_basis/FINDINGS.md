# FINDINGS — version_07_prime_basis

Model: `Qwen/Qwen2.5-0.5B`, CPU, forward passes only. 291 words × 12 carrier
frames = 3,492 passes. Layer selected by split-half reliability: **layer 18**
of 24.

## Verdict: H1 DEAD (earned), H2 emphatically ALIVE

**The NSM primes are not a privileged *spanning basis* for concept vectors in
this model. They are, however, strongly organized as *semantic axes* — NSM's
own category structure is written all over the activation geometry.**

## F1 — measurement: PASS, decisively

| check | value | bar |
|---|---|---|
| median split-half cosine (even vs odd frames) | **0.911** | > 0.5 |
| cross-half top-1 retrieval (291 words) | **1.000** | > 0.8 |

The vectors are real, stable, and uniquely identify every one of the 291
concepts across disjoint template halves. Nothing downstream can be blamed on
measurement noise.

## F3 / H1 — the privileged-basis test: DEAD

OMP reconstruction of 100 target-word vectors from the 61-prime dictionary vs
random 61-word dictionaries drawn from the frequency-matched control pool.
Pre-registered criterion: p < 0.05 at k=8. With the sharper 200-dictionary
null:

| k | primes R² | null R² (200 dicts) | p | per-target win rate |
|---|---|---|---|---|
| 4 | 0.1561 | 0.1476 ± 0.0050 | 0.045 | 0.56 |
| 8 | 0.2184 | 0.2098 ± 0.0057 | **0.075** | 0.53 |
| 16 | 0.2841 | 0.2743 ± 0.0066 | 0.065 | 0.56 |

The frame-confound check (usage frames only) passed — whatever effect exists
is not a mention-frame artifact — but the effect itself is ~0.009 R², the
primes sitting at roughly the 93rd percentile of random common-word
dictionaries. Directionally positive at every k, never significant at the
pre-registered bar. That is not a privileged basis; that is "slightly better
than average common words."

The deeper number is the **absolute** R²: ~0.22 at k=8 for *any* dictionary.
At layer 18 a word's vector is overwhelmingly idiosyncratic — no 61 common
words linearly span more than about a fifth of it. Lexical identity in
activation space is high-rank. The "word = bundle of primes" model fails not
because the primes are the wrong 61 words, but because *no* 61 words span
meaning this way.

## H2 — category structure: ALIVE at p < 0.0001

Within-NSM-category cosine 0.1512 vs between-category 0.0096; statistic
0.1416 against a permutation null of −0.0001 ± 0.0103 (10k shuffles). The
model's geometry recovers NSM's hand-made category system with a z-score of
~14. The top prime pairs are textbook semantic axes:

```
above ~ below   0.822    good ~ bad     0.691    someone ~ something  0.655
before ~ after  0.596    big ~ small    0.491    little ~ small       0.624
```

Near-antipodal contrast pairs, exactly where NSM puts them.

## What this kills, and what it doesn't

**Killed:** the weight-space phase 2 as originally conceived. The pilot was
the gate: if prime *activations* don't privilege-span word activations, prime
*weight-deltas* will not privilege-span word deltas — the ~660-fine-tune
experiment would measure the same null with more zeros on the bill. Do not
run it.

**Survives, reframed:** the primes behave like **axes of contrast**
(above/below, good/bad, before/after), not like dictionary atoms. That is
consistent with how NSM actually works — explications are structured
sentences that *bind* primes into frames, not bags that *bundle* them. Linear
reconstruction only ever tested the bundling half, and the bundling half is
what died. The binding half is precisely what activation/weight arithmetic
cannot express — and what the `version_06_hdc_forward` lattice can (bind ⊗ +
bundle both exist there).

## If there is a version_08 in this line

Not "primes as basis" but "primes as axes": extract *contrast directions*
(v_good − v_bad, v_above − v_below, v_before − v_after, ...) and test whether
target words project onto a small set of prime axes more informatively than
onto random word-pair axes. And if primes are to compose into word meanings
at all, the composition lives in a VSA — bind role⊗prime structures in the
HDC lattice and ask whether *those* predict activation-space neighborhoods.
That is a v06×v07 crossover, not a rerun of this test.

## Reproduce

```
python step1_vectors.py   # ~6 min CPU -> vectors.npz (not committed, ~96 MB)
python falsify.py         # F1 PASS, F2 PASS, F3 FAIL -> HYPOTHESIS DEAD
python step3_structure.py # H2: p < 1e-4
```

Harness honesty was checked both ways on synthetic data before the real run:
planted prime structure → detected at p=0.000; structureless null → not
rejected (p≈0.7, win rate 0.5).
