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

## Teaching the first program (build-time)

```
python prime-lattice/teach.py        # alphabet -> sounds -> form 1000 words -> lexicon.json
python prime-lattice/compounding.py  # show connectivity compounds
```

`teach.py` is the first program: it knows how to take **letters** and their
**sounds** and **form words** in the lattice. Teaching seeds the alphabet (26
primes) and the phonology (digraph + context-aware grapheme→phoneme rules),
then forms 1000 real words, persisting them as `lexicon.json` — the
deterministic db the runtime reads. No model in the loop.

First run: **655/1000 words formed cleanly**; 345 hit a junction where
letters+sounds under-determine the word.

**The point about scale.** Those rules are hand-written stand-ins. The design
is for **LLMs to author the programs/resolutions at build time** (constrained,
then frozen) — the 345 strained words are precisely the LLM's work queue.

**Connectivity compounds.** `compounding.py` proves it: the 345 strained words
collapse to **4 gap-signatures** (≈102 junctions repaired per authored rule).
One LLM-authored resolution (`s`→/z/ between vowels) propagates through every
word sharing that prime. Build-time work is **sublinear** in words; each new
word reuses resolutions already in the lattice. That is the compounding.

## Storing the primitive (schema-on-write, built for 100M+ rows)

```
python prime-lattice/store.py            # demo + benchmark
python prime-lattice/store.py 2000000    # benchmark N rows
```

`store.py` persists the lattice with **schema-on-write**: every record is
validated at the door, and the schema *encodes the architecture's laws* —
a prime may be untyped, anything above the crystallisation line **must** be
typed, every substrate edge must reference an existing row, provenance is
mandatory. Built for scale: **deterministic content-IDs** (idempotent swarm
writes, no coordination), **normalised edges** indexed both ways (O(degree)
traversal), **batched UPSERT** ingest. SQLite prototype, Postgres-ready DDL.
~23k rows/s single-writer; scales ~linearly by hash-sharding writers.

## The build engine + multiple towers

```
python prime-lattice/numbers.py   # math tower: primes that COMPUTE (100% clean)
python prime-lattice/build.py     # spin agents: primes -> programs -> system
```

`numbers.py` is a second primitive tower (digits/operators → numbers →
expressions that evaluate), ~100% deterministic — the cleanest test of
"no probability at runtime" — and it joins the language tower (`"three"` ↔ `3`).

`build.py` is the engine: a generic `primes → programs → system` pipeline that
*any domain falls into*. Agents (LLM calls at scale; deterministic seeds here)
emit each layer, schema-on-write validates them, and they land as typed nodes +
substrate edges in one shared lattice. It builds two domains through the same
pipeline — a **grammar system** (category primes → phrase-rule programs → an
english grammar, rules stored as data) and **math** — proving the structure is
domain-general. Next rungs (reading, comprehension) are the same pipeline.

## One program, all rules (rules as data)

```
python prime-lattice/reader.py    # generic reader interprets grammar from the lattice
```

`reader.py` hardcodes no grammar. It loads every production from the lattice and
parses with a generic Earley engine (any CFG, including left recursion). The
demo parses with the seeded grammar, then a "swarm" adds **one** rule
(`NP → NP conj NP`) as a row — and the *same* program parses a sentence it
couldn't before, with **zero code change**. That is the thesis in one run:
*primes + lattice + agents at scale → the complete deterministic system*, and
every single rule lives in the lattice, not the code.

## Comprehension — the inverse tower (structure → meaning)

```
python prime-lattice/comprehend.py
```

The reader *expands* productions (form → structure); comprehension *inverts*
them (structure → meaning) — compositional semantics. Every grammar production
has a paired **semantic rule** (its inverse), and the inverse pairing lives in
the lattice (each semrule's substrate includes the production it inverts). One
generic interpreter composes meaning bottom-up:
`the big cat saw a dog → SEE(agent=cat[BIG,KIND,LIVE,SMALL,MOVE], patient=dog[…])`.
Built by the same `primes → programs → system` pipeline — a new tower (new
sem-primes + shared primes), same machinery.

## Composition + the brain (controller)

```
python prime-lattice/emotion.py    # affect, on the same NSM primes
python prime-lattice/compose.py    # bind towers into a situation
python prime-lattice/brain.py      # the controller: inherit + dispatch by type
```

`emotion.py` — affect as a tower (NSM emotion explications, shared primes).
`compose.py` — the composition system binds reader+comprehend+emotion into one
situation: *"the dog bit the cat" → cat feels ANGER (because dog BITE cat)* —
capability no single tower has.
`brain.py` — the **controller**. It implements nothing; it **inherits typed
capabilities** advertised by the systems in the lattice and **auto-composes a
pipeline by matching types** (`text→structure→meaning→appraisal→affect`). Add a
system + capability node and the brain inherits the new behaviour with zero code
change. The controller is itself a system in the lattice.

## The whole prototype (one architecture)

```
primes ── programs ── systems ── composition ── controller (brain)
  ▲                                                   │ inherits + dispatches by type
  └──────────── one lattice, schema-on-write ─────────┘
towers built so far: phonics · grammar · math · semantics ·
comprehension(inverse) · emotion · composition · brain
```

## Honest open edges (next stakes)

1. **Promotion criterion** is just a frequency threshold today; real version =
   minimum-description-length (promote when reuse pays for itself).
2. **Strained junctions** are only flagged, not resolved — the choice of
   *rule vs. weight* there is the whole A/B/C learning decision, localised.
3. **Build-time seeding**: the rules/POS are hand-written here; the design is
   for an LLM to author them once, constrained, then freeze.
