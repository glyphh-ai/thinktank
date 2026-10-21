# FINDINGS — version_17

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

## Reproduce

```
python circuits.py      # gates, balance, shortcut floor
python -u run.py        # full sweep (~30 min CPU) -> results.json + gates
python report.py        # re-print tables and gates from results.json
```
