# FINDINGS — version_21_bitspin

One seed, synthetic facts, M2 Max. Logs: `step1.log`, `step2.log`,
`step3_bitspin.log`, `step3_transformer_hi.log`, `step3_transformer_lo.log`.

## step1: bits and counters predict English, about as well as a transformer only while learning

Learn from five novels (4.0 MB), predict 202k chars of *Pride and Prejudice*.

| model | adapting | frozen | size |
|---|---|---|---|
| order-5 n-gram (version_19) | 1.900 | – | – |
| **bitspin** (all heads) | **1.800** | 1.982 | 273 MB of counters |
| bitspin, byte contexts only | 1.783 | 1.971 | 147 MB |
| bitspin, no soft (angle) heads | 1.801 | 1.983 | 231 MB |
| byte transformer, 10.9M params, 8000 AdamW steps | – | **1.773** | 44 MB |

**Read:**
- With no backprop and no real-valued weights apart from 72 mixing
  numbers, bitspin beats the n-gram by 5%. It ties the transformer only
  when it keeps learning while it reads; frozen it's 12% worse, at 6× the
  memory.
- **The angle part did nothing.** Soft heads (majority-bundled contexts
  looked up by angle) changed the score by 0.001. What worked was exact
  XOR binding plus counters. The word and memory heads cost 0.017 here;
  they're there for step3.

## step3: one showing teaches a fact, a correction moves only that fact

300 invented facts ("Piheba Doti keeps a hawk named Nipira."), shown 1–3
times; then the first 100 get a new name, shown 1–3 times. Recall = the
right name scores best of 21. Damage = bits/char on 50k chars of *Pride
and Prejudice*.

| model | recall after 1 / 3 showings | corrected after 1 / 2 / 3 | untouched facts kept | P&P before → end |
|---|---|---|---|---|
| **bitspin** (counters only learn) | **100% / 100%** | 0% / **100%** / 100% | **100%** | 2.064 → **2.069** |
| transformer, lr 1e-6 ×1 step | 5% / 6% | 9% / 9% / 9% | 8% | 2.044 → 2.108 |
| transformer, lr 1e-5 ×1 | 6% / 7% | 4% / 10% / 13% | 6% | 2.044 → 2.417 |
| transformer, lr 3e-5 ×3 (its best recall) | 5% / 18% | 15% / 40% / 74% | 14% | 2.044 → 3.044 |
| transformer, lr 1e-4 ×1 | 6% / 7% | 10% / 20% / 55% | 6% | 2.044 → 4.941 |
| transformer, lr 1e-3 ×3 | 5% / 6% | 15% / 8% / 14% | 6% | 2.044 → 9.321 |

**Read:**
- **The claim holds on this test.** Bitspin learned all 300 facts from a
  single forward pass each, took every correction on the second showing,
  kept every fact it wasn't told to change, and English got 0.005 bpc
  worse. No transformer setting got past 18% recall, and every setting
  that learned anything damaged English by 0.4–7 bpc. It's catastrophic
  forgetting, measured.
- **Where the forgetting lives.** An earlier bitspin, where the long word
  context was trusted through the shared mix, learned the facts in two
  showings but pushed English from 2.12 to 3.24 bpc. That's the same
  disease as the transformer, in the one shared part. The fix was to give
  the shared part nothing to learn: the memory head refines the final
  guess by its own counts alone (`p = (n1 + p) / (n + 1)`). A fact then
  lives only in its own counters.
- Corrections take two showings, not one. After one, the old and new
  counts tie at the first letter where the names differ; the hard fade
  (old count halved on contradiction) breaks the tie on the next showing.

## What this does not show (yet)

- **The memory head fits the template.** It binds the last 6 words, and
  the answer sits exactly 6 words after the owner's first name. A
  paraphrased question ("What is Piheba Doti's hawk called?") isn't
  tested. Generalizing across phrasings is the job the angle (soft) heads
  were meant to do, and in step1 they did nothing.
- The transformer got plain online Adam. Replay, LoRA or a memory
  module (Titans/TTT) would forget less. That's the fairer next opponent.
- Memory: 273 MB of counters vs a 44 MB transformer that is better frozen.
- One seed, synthetic facts, recall among 21 candidates rather than free
  generation; whole-page bit diffusion isn't built.

## Next

1. Paraphrase test: teach in one phrasing, ask in another. This is where
   angle-matched (soft) memory has to earn its place.
2. Against a transformer with replay or a TTT memory layer, at equal bytes.
3. Free generation of the answer (greedy bits), not multiple choice.
