# FINDINGS — version_09_semantic_barcode

Frozen `Qwen/Qwen2.5-0.5B`; 1027-word eval vocabulary (`wordlist.txt`);
blind layer criterion picked **layer 5** (split-half reliability 0.915;
layer 18 was 0.908 — a near-tie, and the exploratory layer-18 rerun below
shows the verdict is not layer-luck). All criteria were committed before any
barcode existed (`e128d01`).

## Verdict: FAITHFUL BUT NOT COMPOSITIONAL — with the best surprise of the line so far buried in K2

| gate | result |
|---|---|
| K1 width (≥256 stable axes) | **PASS** — 787/1830 prime-pair axes sign-stable ≥0.90 |
| K2 fidelity (tax ≤20% at K=256) | **PASS** — tax **4.7%**, and *negative* below K=256 |
| K3 composition (≥7/15 probes) | **FAIL by one** — 6/15, null 0.42±0.60, p=0.000 |
| K4 lattice (recovery + query) | **split** — bit recovery **1.0000**; query undefined (see below) |

## K2 — the fidelity-tax curve is the headline

Spearman ρ between code agreement and full 896-dim cosine geometry, all
~527k word pairs:

| K | named (prime pairs) | SimHash | random word pairs | tax |
|---|---|---|---|---|
| 16 | **0.311** | 0.197 | 0.307 | **−58%** |
| 32 | **0.413** | 0.258 | 0.415 | **−60%** |
| 64 | **0.474** | 0.360 | 0.503 | −32% |
| 128 | **0.547** | 0.473 | 0.566 | −16% |
| 256 | 0.584 | 0.613 | 0.619 | +4.7% |
| 512 | 0.592 | 0.730 | 0.654 | +19% |

Three facts, in decreasing order of importance:

1. **Below K≈256 the tax is negative: named bits BEAT SimHash.** At K=32 a
   named barcode carries ~60% more geometry than the LSH optimum-in-the-limit.
   Data-informed axes (differences of real word vectors) concentrate on
   directions where words actually vary; random hyperplanes in 896-d mostly
   sample low-variance space. In the regime that matters for a compact,
   auditable code — tens to a couple hundred bits — **interpretability is
   free, and then some.**
2. **The crossover is real.** Pair-axes saturate (787 correlated axes to draw
   from) while SimHash keeps adding independent bits; by K=512 random wins by
   19%. If you need maximal fidelity at large K, use SimHash; if you need
   bits that mean something, the named code costs nothing until ~256.
3. **Prime-ness is still not the magic — pair-ness is.** Random control-word
   pairs track (even slightly beat) prime pairs at every K, third v09-v08-v07
   result in a row saying the NSM inventory is not privileged in this model.
   The value of primes here is that their axes come pre-*named* and
   human-sensible, not that they carry more signal.

## K3 — composition: one probe short, massively above chance

6/15 pre-registered bundles hit top-20 of ~1025 (null: 0.42 ± 0.60 hits,
p=0.000). The hits are exactly what the idea promised — `young+woman → girl`
(rank 4), `royal+man → prince` (4), `money+place → market` (5),
`school+person → student` (5), `big+water` (16), `water+sky` (17). The misses
share one failure mode: the eval vocabulary is a *web-frequency* list, and
layer-5 neighborhoods are register-dominated — film/jobs/videos/software crowd
out `ice` (rank 42), `soldier` (42), `boy` (83). Per pre-registration this is
a FAIL, and it stays a FAIL; but the mechanism (bundle = feature
intersection) demonstrably retrieves lexicalized concepts far above chance.

Exploratory layer-18 rerun (`exploratory_layer18.py`, labeled, not
pre-registered): 571 stable axes, tax 7.0% at K=256, composition **2/15** —
worse. Layer 5 was the right pick; mid-early layers compose better here.

## K4 — the lattice is fine; the *bit* definition is the bug

Bit recovery from bound records is perfect: 1027 words × 256 bits stored as
one D=8192 bipolar record each (`Σ bitᵢ·axis_idᵢ`), read back by unbinding at
**100.00%**. The carrier works at real width, exactly as v08's H5 predicted.

The feature query failed for a reason worth the whole version: `good-bad` —
v08's best meter, AUC 0.964 — has sign-stability only **0.875** on this
vocabulary, under the 0.90 bar (layer 18: 0.869; `someone-something` passes
at both layers). No contradiction: AUC certifies *ordering at the extremes*;
a barcode bit demands a *stable sign everywhere*, and most of a 1000-word
vocabulary is nearly neutral on valence — near-zero projections flip signs
between template halves. **A meter is not automatically a bit.** Forcing the
neutral mass to pick a side is the design flaw.

## What v10 should be (the data says it loudly)

**The ternary barcode**: bits in {−1, 0, +1} with a certified dead zone —
words abstain on axes where they're neutral. This directly fixes the K4
failure mode (neutral words stop flipping), should *raise* fidelity per bit
(sign noise near zero is exactly what dilutes K2 at large K), makes queries
sharper ("valence = −1" stops matching accidental negatives), and likely
rescues several K3 probes (bundle agreement concentrated on committed bits
instead of diluted by coin-flips). Sparse ternary is also the native
regime of real VSA memories. Same pre-registration discipline: fix the dead
zone by stability, not by outcome.

## Reproduce

```
python vocab.py                  # validate committed wordlist + probes
python step1_vocab_vectors.py    # ~12 min CPU -> vocab_vectors.npz (~217 MB)
python falsify.py                # K1 PASS, K2 PASS (tax 4.7%), K3 FAIL 6/15, K4 split
python exploratory_layer18.py    # labeled exploratory: layer 18 is worse
```
