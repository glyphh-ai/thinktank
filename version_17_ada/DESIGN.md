# version_17_ada — DESIGN: typed atomic construction

## What the experiments taught us

| attempt | what happened |
|---|---|
| v13 NSM lattice, values = random atoms | structure stored, but similarity thin; 35% of words share a code |
| closed sets (values are glyphs) | converges, but NSM primes define themselves — no new meaning |
| open-vocab closure (v1/v2/v3) | 94-99.6% of roles decode exactly: **glyphs store structure faithfully** |
| lattice decode (energy spread) | "love is between people" emerged from two threads converging — said by no single role |
| context fire | "mean" split into cruel / intend by context alone — **senses are derived, not stored** |
| every attempt to put meaning INTO the glyph (weights, `none`, branches, padding) | made it worse |

**Glyphs carry structure. Meaning is derived.** The LLM knows meaning but has no
structure; the glyph has structure but no construction. What was missing is
**atomic construction**: every part of the schema is itself a glyph whose meaning
includes *how it binds*.

## The anatomy: typed atoms at every level

Every layer, segment and role is a word with its own glyph (closed set), and
each carries a **type** that decides how it composes.

### Roles — typed association (how a value attaches)

| type | roles | binding |
|---|---|---|
| attach | class, number, tense, degree, slot, register, mood, feeling, goodness, strength, place, time | `K ⊙ g(v)` — stored, decodable |
| inherit | base, category | `K ⊙ g(v)` **+** pull toward `m(v)`: the word inherits its parent's meaning |
| attract | similar, partner, figure, phrase | `K ⊙ g(v)` **+** pull toward `m(v)` |
| negate | opposite | `K ⊙ g(v)` **−** pull: the word is pushed *away* from its opposite |
| contain | part, whole | `K ⊙ ρ^±1(g(v))` — containment is directional (part ≠ whole) |
| order | before, after | `K ⊙ ρ^∓2(g(v))` — position as permutation: "i love" ≠ "love i" |
| direct | cause, purpose, agent, patient | `K ⊙ ρ^±3(g(v))` — one-way: cause flows in, purpose flows out |

`K` = path key (layer ⊙ segment ⊙ role glyphs), `ρ` = cyclic permutation (exactly
invertible), "pull" = an *unkeyed* contribution in meaning space — it moves where
the word sits instead of being a fact stored in it.

### Segments — typed composition (how roles combine)

| type | segments | composition |
|---|---|---|
| bundle | kind, form, tone, sense, relation, context, expression | sum of role terms — independent facts |
| together | frame | sum **+** one product term `agent ⊙ ρ(patient)`: the event exists only as a whole |
| sequence | position | `before → slot → after` as permutation-ordered terms |

### Layers — stacked (how levels relate)

`grammar → sentence → idiom → meaning`. Each layer is kept as its own keyed vector
(peelable: `cortex ⊙ K(layer)`), the cortex is their sum. **Conditioning between
layers (idiom overrides meaning, grammar constrains sentence) is the job of the
reader**, not the store — it is context-dependent, so it belongs to the
interpretive model (`context_fire.py` is its hand-built seed).

## Division of labour

- **LLM**: chooses which atoms fill a word's roles (meaning it already knows).
- **Schema anatomy**: decides how atoms are assembled (structure the LLM lacks).
- **Interpretive model** (next): reads glyphs in context and derives meaning —
  energy routing through the typed lattice, settled by context. Structure in,
  meaning out, same lattice shape.

## Tests (pre-registered, `typed_encode.py`)

Typed vs untyped (every role `attach`, today's v3 encoding), same glyphs:

1. **Opposites point away**: cos(meaning layer) of love/hate and kind/mean
   becomes **negative** typed (untyped: positive).
2. **Synonyms stay close**: love/care and kind/nice stay positive.
3. **Structure survives**: role decode accuracy stays ≥ 90% (untyped 94%).
4. **Direction is real**: decoding `before` returns the before-word, not the
   after-word, and `part` ≠ `whole` (order/contain types are recoverable).
