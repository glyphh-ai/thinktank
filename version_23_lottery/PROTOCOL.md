# PROTOCOL — version_23_lottery (written 2026-11-22, before any model ran)

> **Can a think-tank language model predict Texas Lottery winning numbers
> better than chance?**

Expected answer: no. The test is built so that a "no" means something:
a positive control (a planted pattern) must be found, and a negative
control (a true random stream) must not be.

## Data

Texas Lottery CSV exports (texaslottery.com, all years, pulled
2026-11-22, last draw 2026-11-21), in `data/`. Each game is kept only
under its current rules, so every draw is scored against the same pool:

| game | rules scored | from |
|---|---|---|
| Lotto Texas | 6 of 54, draw order | 2006-05-01 (6/54 returned Apr 2006) |
| Cash Five | 5 of 35, draw order | 2018-09-24 (pool shrank from 37 in Sep 2018) |
| Texas Two Step | 4 of 35 + bonus 1 of 35 | all (2001-05-18) |
| Pick 3 | 3 digits, 0–9, all four daily draws in time order | all (1993) |
| Daily 4 | 4 digits, 0–9, all four daily draws in time order | all (2007) |

Fireball digits are ignored. Each draw becomes one line of text, one
byte per number (digits as '0'–'9', ball n as `chr(64 + n)`, so 1 is
'A'), ended by a newline. Only the number bytes are scored.

## Split

Chronological per game: first 70% train, next 10% validation (only for
early stopping / settings), last 20% test. Nothing about the test draws
is looked at before the final run.

## Models

1. **Uniform** — the truth if the draw is fair. Ball k of a draw:
   log2(pool − k + 1) bits; a digit: log2(10) bits.
2. **Hot numbers** — the folk theory: online counts per number with a
   decay, prior strength and decay picked on validation. Predict, then
   count.
3. **Bitspin** (version_21, `bitspin.py` unchanged, default heads) —
   counters, learns on every forward pass, keeps learning through
   validation and test (predict each byte, then count it). Its
   probability spreads over all 256 bytes and is NOT renormalized to the
   legal numbers — this only hurts it.
4. **Byte transformer** (version_21 `ByteLM` architecture, small: d=128,
   4 layers, context 256 bytes ≈ 36–64 draws) — AdamW on train, early
   stopped on validation, frozen on test. Its probability is
   renormalized over the legal numbers (no repeats within a draw).

## Metric

Per test draw: bits(uniform) − bits(model), summed over the draw's
numbers. Positive = the model knew something.

## Pass bar (a model "found a pattern" in a game only if all hold)

1. Mean gain > 0 with one-sided p < 0.01 / 15 (z test on the per-draw
   gains; 5 games × 3 models, Bonferroni), and
2. the mean gain is > 0 in both the first and the second half of the
   test draws, and
3. **the money test**: the model's single most likely ticket for each
   test draw wins more than chance with one-sided binomial p < 0.01 / 15
   (Pick 3 / Daily 4: exact straight match; ball games: count of matched
   main balls versus its expectation under chance).

## Controls (same pipeline, same models, same bar)

- **Negative**: a Pick-3-sized stream from numpy's PCG64. Every model
  must fail. If one passes, the pipeline is broken and nothing counts.
- **Positive**: a Pick-3-sized stream where, 30% of the time, a draw's
  first digit copies the previous draw's last digit (otherwise uniform:
  0.37 bits per draw of real signal; the best ticket wins 0.37% of
  draws instead of 0.10%). The learning models (bitspin,
  transformer) must pass. If neither does, a "no" on the real games
  says nothing.

## What gets reported

Every game × model row, pass or fail, with gain in bits per draw, p,
and money-test hits versus chance, in FINDINGS.md. No re-runs with new
settings after seeing the test.

---

# Addendum A — HDC model with beam-search tickets (written 2026-11-22, after run 1, before HDC ran)

Run 1 is done (`run.log`, `results.json`): no model passed any game,
the negative control failed everywhere, **and the positive control also
failed for bitspin and the transformer**, so their "no" says nothing.
Run 1 is reported as is; nothing in it is re-run or retuned here.

Asked next (Chris): an HDC model, with beam search choosing the tickets.

## Model 5: HDC (version_06 `hdc.py`, bipolar, D = 10,000, unchanged)

- **Context** for number j of a draw: the majority bundle of
  `bind(role(lag k, position i), value(v))` for every number of the last
  K draws, plus `bind(role(this draw, position i), value(v))` for the
  numbers already drawn in this draw.
- **Memory**: one prototype per (slot, value): the running int sum of
  the contexts that came before that value. Digits: a slot per position.
  Balls: every main ball shares one slot (any position), the bonus has
  its own. Predict, then add (online through train, validation, test).
- **Probability**: cosine of the context with each legal value's
  prototype (prototypes centred on their mean), standardized across the
  legal values to z, then p ∝ exp(β·z) over the legal values.
- **Settings** picked on validation only: K ∈ {1, 2, 4},
  β ∈ {0, 0.01, 0.03, 0.1, 0.3, 1, 3}. β = 0 is uniform, so the model
  may choose to know nothing.

## Tickets by beam search

Width 16: each partial ticket is extended by every legal next number,
scored by the HDC log p given that ticket's own numbers so far, and the
16 best joint scores are kept. The ticket bought is the best full one.
Secondary, digits only: the 10 best beam tickets bought per draw, wins
versus 10× chance (binomial).

## Pass bar

The same three conditions, with α = 0.01 / 20 (the 15 run-1 tests plus
these 5). Controls run first in the same pipeline: HDC must fail the
random stream and pass the planted one, or its result on the real games
says nothing.
