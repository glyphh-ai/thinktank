# FINDINGS — version_14

The electrical-engineering inversion: what if multi-head attention doesn't
need learned diversity — just geometric coverage of the position space?
Replace learned positional embeddings with deterministic π-frequency
sinusoidal encodings, then split attention into 3 phases (0°, 120°, 240°)
sharing the same Wq/Wk/Wv/Wo weights. Like a 3-phase transformer in
power systems — no positional blind spots, constant coverage at every
frequency.

## Design — 3-phase π-frequency attention

Position encoding: `freq = 1/π^(2i/C)`, giving wavelengths from 2π (~6
positions) to ~2π² (~20 positions) — matched to T=32 context.

3-phase: three copies of the encoding at phase offsets 0, 2π/3, 4π/3
(the equilateral triangle from the law of sines with side π). Each
phase adds its positional encoding to the layernormed token embeddings,
computes Q/K/V through the *same* weight matrices, runs causal
attention, and produces an output. The three outputs are averaged and
projected through Wo.

Key property: the heads are differentiated by *geometry*, not by learned
projections. Zero parameters spent on position — the entire position
representation is deterministic.

## Strat 0 — progressive gates

### Gate 0: single-phase π-pos vs learned pos (v0 baseline)

Same model, same data, same seeds. Only change: replace the learned
pos_emb (32×128 = 4,096 parameters) with deterministic π-frequency
sinusoidal encoding (0 parameters).

| embedding       | val loss | val acc | params    |
|-----------------|----------|---------|-----------|
| learned pos     | 3.12     | 35.0%   | 1,583,487 |
| single-phase π  | **3.13** | 35.5%   | 1,579,391 |

**PASS** — π-pos matches learned pos with 4,096 fewer parameters.
Position information at T=32 does not need to be learned.

### Gate 1: 3-phase π-pos vs single-phase π-pos

Same model, same data, same seeds, same weight count. Only change: the
single attention computation becomes three, each with a different phase
of the positional encoding (0°, 120°, 240°), outputs averaged.

| embedding         | val loss | val acc | best val loss |
|-------------------|----------|---------|---------------|
| single-phase π    | 3.13     | 35.5%   | 2.917         |
| **3-phase π**     | **3.08** | **36.8%** | **2.875**   |

**PASS** — 3-phase beats single-phase with zero extra parameters.
The improvement comes purely from geometric coverage of position space.

### Gate 2: 3-phase π-pos vs learned pos (v0 baseline)

| embedding       | val loss | val acc | params    |
|-----------------|----------|---------|-----------|
| learned pos     | 3.12     | 35.0%   | 1,583,487 |
| **3-phase π**   | **3.08** | **36.8%** | 1,579,391 |

**PASS** — 3-phase π-pos beats learned position with fewer parameters.
Best-ever val loss for this model (2.875 at step 7000).

## What this means

1. **Multi-head behavior from geometry alone.** The three phases create
   three distinct attention patterns using shared weights. No per-head
   projections, no learned specialization — the diversity comes from
   positional phase offsets.

2. **Position is not a learning problem** (at this scale). The model
   doesn't benefit from learning position — it benefits from having
   position *specified correctly* from construction.

3. **The tradeoff that breaks.** Standard multi-head attention forces
   more heads = smaller per-head dimension (d_model / num_heads).
   Phase attention scales head count without shrinking head capacity.
   Each phase operates on the full embedding dimension.

4. **Separation of concerns.** Weights handle content ("what attends
   to what"). Phases handle position ("where things are"). These are
   cleanly separated — the same weights could transfer across different
   context lengths or phase configurations.

## Open questions for v15+

- **Does the curve keep going?** 6-phase, 12-phase, 36-phase — does
  more phases = more improvement, or does it saturate?
- **Phase spacing.** The 120° offset comes from the 3-phase transformer
  analogy. Is equidistant spacing optimal, or do other arrangements
  (e.g., golden-ratio spacing) work better?
- **Interaction with the v13 lattice.** The π-pos handles position;
  the NSM lattice handles semantics. Combining them: authored position
  + authored semantics + learned content processing.
- **Scale.** This is C=128, T=32, 1 block. The hypothesis is that the
  gap widens at scale — more dimensions, longer context, deeper models.

## Reproduce

```
python test_train.py                  # gradient check + 700-step gate
python -u strat0_train.py             # full 8k-step run (~45 min CPU)
```
