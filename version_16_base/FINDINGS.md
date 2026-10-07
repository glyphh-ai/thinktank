# FINDINGS — version_16_base

## Strat 0 — the new base speaks by step 5,000: PASS

4 blocks × 4 heads, C=256, T=64, 5.02M params, hand-written backprop
(finite-difference gradient check: worst relative error 6e-5). 5,000 AdamW
steps, B=32 × T=64 = 10.2M tokens, < 1 epoch of the 24M-token train slice;
val is TinyStories-valid, never trained on. ~1 h on a 4-core CPU.

| model | params | steps | val loss | val acc |
|---|---|---|---|---|
| v0 base (1 block, 1 head) | 1.58M | 8,000 | 3.12 | 35.1% |
| v15 best (12-head) | 1.58M | 8,000 | 3.00 | 37.4% |
| **v16 base** | **5.02M** | **5,000** | **2.56** | **45.5%** |

(v16 vocab is 7,218 vs 5,375, so its uniform loss is higher: 8.88 vs 8.59;
the gap is still large. Final number: 40 batches × 64 × 64 tokens.)

It passed the old base's final val loss (3.12) at step ~1,250. Train/val gap
at step 5,000: 0.05 — no overfitting; more steps would keep helping.

Sample (seed 1, temperature 0.8, top-k 40):

> once upon a time, there was a little boy named timmy. timmy loved to play
> outside in the sun. one day, timmy saw a big tree with a long trunk. [...]
> " mommy, can we go to the tree? " timmy asked. " sure, sweetie. the tree is
> nice to be safe, " his mommy replied. " but it's important to be careful and
> not to play with the leaves. [...]"

Grammatical, dialogue opens and closes, characters persist across sentences.
Remaining failures are semantic: a flower "in the sky", a character name
switching mid-story (amy → sue), "swing and forth".

Full curve: `strat0_history.json`; full log + 3 samples: `strat0_log.txt`.
