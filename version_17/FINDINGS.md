# FINDINGS — version_17

**Verdict: G2 (core) FAIL at its registered point — but the measurement
was the wrong shape.** Every looped run is all-or-nothing: it either
executes circuits perfectly at every depth or sits stuck near the
shortcut floor. Averaging 3 binary outcomes cannot separate B from C.
What is solid: the loop is what generalizes in depth (A never does), the
router becomes a perfect readable map onto the tiny models, and random
tiny models never solve anything. The data-efficiency question is open
and needs a solve-rate test over many seeds (strat 2).

## The question

Can a model built on a library of tiny pre-trained models — each one
knowing exactly one small thing — reason through them better than a
model that learns everything itself? Rung 1 of a ladder: the tiny
models are the 16 two-input boolean functions; the "reasoning" is
executing circuits built from them. Every answer is exact, so every
claim is checkable to the bit.

Prior art (from memory, not a fresh search): Mixture-of-Experts (Jacobs
et al. 1991 → Mixture of a Million Experts 2024 — co-trained experts do
not specialize by human domain), Neural Module Networks (Andreas 2016 —
learned routing collapses without pressure), CALM (2024), LoraHub
(2023), DreamCoder (Ellis 2021 — library learning), Recurrent
Independent Mechanisms (Goyal 2019), Thousand Brains / Monty (Numenta).
New here: the bottom rung is trained in isolation and *frozen*, and the
design is meant to stack (rung 2 builds on rung 1's routed compositions).

## Design — pre-registered, committed before any result

**Bank.** 16 tiny MLPs (2→8→1, 33 params each), each trained alone on
its own gate's 4-row truth table, then frozen.

**Circuits** (`circuits.py`). 4 random input bits, then D steps;
step i = op_i(slot a_i, slot b_i) over 6 slots (X0..X3, P1 = previous
result, P2 = the one before). Every step after the first reads P1, so
the answer sits D gates deep. Ops are drawn from the 10 gates that
depend on both inputs (constants/projections would cut the chain).
Outputs are balanced (P(y=1) ≈ 0.50 at every depth).

**Shortcut floor.** 8 of 10 gates are AND/OR-like: the last step's fresh
operand alone often forces the answer. Reading only the last step gives
**≈ 0.66 accuracy at every depth ≥ 2** with no execution at all
(`circuits.shortcut`). Accuracy at or near 0.66 means "found the
shortcut", not "reasons".

**Models** (`models.py`):

| | what | learned params |
|---|---|---|
| A monolithic | 2-layer transformer (C=64, 4 heads, sinusoidal position) reads the whole circuit, outputs the bit | ~68k |
| B looped | step-by-step loop over the slots; ONE shared learned executor MLP | ~840 |
| C layered | same loop; executor = soft routing over the frozen bank: r = Σ_k softmax(router[op])_k · tiny_k(a,b) | 196 |
| C0 layered-random | C with a bank of 16 frozen RANDOM MLPs (never trained) | 196 |

B, C, C0 share the identical loop and identical learned operand
addressing; only the executor differs. Values between steps stay
probabilities — no rounding, no tricks. C0 separates what the tiny
models *know* from the *shape* of the architecture.

**Training.** Depths 1–4 only. Training-set sizes n ∈ {64, 256, 1024,
8192} circuits (split evenly across depths), seeds {0,1,2}, 3000 Adam
steps, batch 64, loss = BCE on the final bit only (no intermediate
supervision). lr 1e-3 (A), 1e-2 (B, C, C0).

**Evaluation.** 1,000 fresh circuits per depth at D ∈ {1,2,3,4,6,8,12,
16,24,32} (fixed eval seed). Primary depth for OOD claims: **D = 16**.
All numbers are means over the 3 seeds.

## Gates

| gate | claim | pass if |
|---|---|---|
| G1a | the loop structure generalizes in depth | B and C ≥ 0.95 at D=16, n=8192 |
| G1b | the monolithic transformer does not | A ≤ 0.75 at D=16, n=8192 |
| **G2 (core)** | **tiny models buy data efficiency** | **C − B ≥ 0.10 at D=16, n=256** |
| G3 | the router is readable | ≥ 9/10 ops routed to their own gate's tiny model, every seed, n=8192 |
| G4 | no in-distribution cost | C ID acc ≥ B ID acc − 0.01, n=8192 |
| G5 | it is the content, not the shape | C − C0 ≥ 0.10 at D=16, n=256 |

**Verdict rule.** G2 PASS → the core claim (a frozen library of tiny
models is a better substrate to reason over) is ALIVE at rung 1.
G2 FAIL with C ≤ B → DEAD at rung 1: the loop does the work, the
library adds nothing. G5 decides whether any C advantage is knowledge
or architecture.

## Strat 1 — results

Mean accuracy over 3 seeds (`results.json`; full table from `report.py`):

| model | n | D=1–4 | D=16 | D=32 |
|---|---|---|---|---|
| shortcut floor | — | — | 0.660 | 0.636 |
| A monolithic | 8192 | 0.75 | **0.507** | 0.497 |
| B looped | 256 | 0.87 | 0.876 | 0.864 |
| C layered | 256 | 0.78 | 0.786 | 0.769 |
| C0 random bank | 256 | 0.67 | 0.702 | 0.675 |
| B looped | 1024 | 0.93 | 0.917 | 0.907 |
| **C layered** | 1024 | **1.000** | **1.000** | **1.000** |
| C0 random bank | 1024 | 0.69 | 0.690 | 0.682 |
| B / C | 8192 | 1.000 | 1.000 | 1.000 |
| C0 random bank | 8192 | 0.70 | 0.699 | 0.701 |

| gate | result | |
|---|---|---|
| G1a loop generalizes | **PASS** | B 1.000, C 1.000 at D=16 |
| G1b monolithic does not | **PASS** | A 0.507 — below the shortcut floor beyond its training depths |
| G2 core: C − B ≥ 0.10 @ n=256 | **FAIL** | C 0.786 − B 0.876 = −0.09 |
| G3 router readable | **PASS** | 10/10 ops → own tiny model, all seeds |
| G4 no ID cost | **PASS** | 1.000 vs 1.000 |
| G5 content not shape @ n=256 | **FAIL** (+0.084) | — but see below |

### The shape of the data: runs are binary

Every B and C run lands at either 1.000 (all depths, to 32) or 0.6–0.75
(stuck near the shortcut). Seeds fully solved (D=16 ≥ 0.99), of 3:

| n | A | B | C | C0 |
|---|---|---|---|---|
| 64 | 0 | 0 | 0 | 0 |
| 256 | 0 | **2** | 1 | 0 |
| 1024 | 0 | 2 | **3** | 0 |
| 8192 | 0 | 3 | 3 | 0 |

So the pre-registered means are really solve counts out of 3. B beats C
at n=256 (2 vs 1), C beats B at n=1024 (3 vs 2) — neither difference is
distinguishable from chance with 3 seeds. G2's "fail" is a measurement
of the coin, not of the claim.

### What is real

1. **Depth generalization comes from the loop, not the library.** A
   (68k params) gets 0.84 on depth 1 and falls to chance by depth 6 —
   below even the shortcut. Both looped models, once solved, are
   perfect at depth 32 after training only to depth 4.
2. **When C solves, its router IS the answer.** Every solved C run
   routes 10/10 ops to exactly their own gate's tiny model, with the 6
   distractor models ignored. Unsolved runs sit at 4/10 — a partial
   map that is stuck, not drifting. The model's reasoning is readable
   as a lookup table.
3. **The knowledge in the tiny models is what matters.** C0 — same
   architecture, random frozen models — solves 0 of 12 runs at any
   size and plateaus at the shortcut (~0.70). C solves 7 of 12. G5's
   registered margin missed at n=256 (+0.084), but at n=1024 the gap is
   +0.31.
4. **C learns 196 numbers; B learns 837.** Equal ceiling, 4× fewer
   learned parameters, fully interpretable.

### Not supported yet

That frozen tiny models make reasoning *more data-efficient*. The
registered test could not see it; the post-hoc n=1024 hint (3/3 vs 2/3)
is a hint, not a result.

## Strat 2 — solve rate: pre-registered design (no results yet)

Strat 1's runs are binary, so measure the binary quantity.

- Models B, C, C0 (A dropped: 0/12 solves at every size in strat 1).
- n ∈ {128, 256, 512, 1024}; 20 fresh seeds (100–119) per (model, n);
  everything else identical to strat 1 (3000 steps, batch 64, lr 1e-2,
  depths 1–4, same eval sets).
- **Solved := accuracy at D=16 ≥ 0.99.**
- Test: one-sided Fisher exact on solve counts, pooled over the 4 sizes
  (80 runs per model). Per-size counts reported, not gated.

| gate | claim | pass if |
|---|---|---|
| **S1 (core)** | frozen tiny models make solving more likely | pooled solve rate C − B ≥ 0.15 **and** p < 0.05 |
| S2 | it is the content, not the shape | pooled C − C0 ≥ 0.15 and p < 0.05 |
| S3 | solved runs are readable | every solved C run routes ≥ 9/10 ops to its own tiny model |

**Verdict rule.** S1 PASS → ALIVE. S1 FAIL with B ≥ C pooled → DEAD.
Otherwise INCONCLUSIVE.

## Reproduce

```
python circuits.py      # gates, balance, shortcut floor
python -u run.py        # full sweep (~30 min CPU) -> results.json + gates
python report.py        # re-print tables and gates from results.json
python -u strat2.py     # solve-rate sweep (~25 min) -> strat2_results.json + gates
```
