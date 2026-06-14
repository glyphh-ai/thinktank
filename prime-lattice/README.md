# prime-lattice

A tiny, runnable stake-in-the-ground for the two-regime / interface-lattice
architecture we sketched. Deterministic, no model in the loop at runtime.

```
python prime-lattice/kernel.py
```

## The one claim it tests

**Type at the level where function stabilises.** Primes (alphabet/sounds) stay
fungible and ambiguous — a cell is a cell. Programs (words/sentences) are typed
by function and crystallise above the line — a heart is a heart. Ambiguity is
not a bug to be compiled away; it *localises* below the crystallisation line.

## Where each idea lives in `kernel.py`

| Idea from the thread | In the code |
|---|---|
| neuron = **interface**, not implementation | `Node` protocol: `receive(Signal) -> Signal` |
| same atom → many parents = **lattice, not tree** | a `Letter` feeds the phonetic pathway *and* the word/orthographic pathway |
| **primes** = alphabet/sounds (fungible cells) | `Letter`, `route_sound` |
| ambiguity is correct at the prime level | `route_sound` returns a `strained` flag instead of forcing a type |
| **programs typed by function** (organs) | `Word.pos`, `Sentence` |
| programs **retain their substrate** | `Word.sound_out()` re-runs the prime pathway over its own letters |
| the **learning knot** = a promotion rule | `Lexicon.observe` (MDL / library-learning style reification) |
| composite stands in for a leaf (Liskov) | `Word` and `Letter` share `receive` |

## The finding from the first run

Deterministic routing resolved every junction **except `g` before a front
vowel** (`gem`=/j/ vs `get`=/g/) — flagged automatically. That is the exact
coordinate where a rule isn't enough and a weight/lookup is required: the
empirical answer to "where, if anywhere, is probability irreducible." It sits
*below* the type line, exactly as the architecture predicts.

## Honest open edges (next stakes)

1. **Promotion criterion** is just a frequency threshold today; real version =
   minimum-description-length (promote when reuse pays for itself).
2. **Strained junctions** are only flagged, not resolved — the choice of
   *rule vs. weight* there is the whole A/B/C learning decision, localised.
3. **Build-time seeding**: the rules/POS are hand-written here; the design is
   for an LLM to author them once, constrained, then freeze.
