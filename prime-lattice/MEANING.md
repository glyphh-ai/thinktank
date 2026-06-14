# Meaning & Memory — structure

The same ladder (primes → programs → systems → situation), plus the one
distinction everything else hangs on: **TYPE vs INSTANCE**.

- TYPE-level nodes are taught once and shared forever (a cat in general, the
  BITE event in general). This is **semantic memory** — the permanent lattice.
- INSTANCE-level nodes are created during a conversation (this cat, this bite,
  right now). This is **episodic memory** — situations and entities, transient.

Generation coherence and memory both fall out of getting this split right.

## 1. MEANING

### Primes  (level = prime, fungible, untyped)
| kind | examples | role |
|---|---|---|
| `sem-prime` | I, KNOW, WANT, FEEL, GOOD, BAD, DO, HAPPEN, SEE, SAY, BIG, NOT, CAN, BECAUSE, IF | NSM atoms of meaning |
| `role` | AGENT, PATIENT, EXPERIENCER, RECIPIENT, LOCATION, TIME, CAUSE, INSTRUMENT | who/what slots |
| `logic` | AND, OR, NOT, IF-THEN, BEFORE, AFTER | composition / inference glue |

### Programs  (level = program, typed by function) — four layers
1. **concept** (a word-sense) — `substrate = [sem-primes]`
   `concept:cat#1 → {KIND, LIVE, SMALL, MOVE}` (homonyms = multiple concepts per word)
2. **frame** (a predicate + its slots + **selectional restrictions** + entailments)
   `frame:BITE → slots{agent:[ANIMATE], patient:[PHYSICAL]}, entails: FEEL(patient, BAD)`
   *This layer is what fixes word-salad: it says who-can-do-what-to-what.*
3. **proposition** (a frame with slots filled) — `substrate = [frame, fillers]`
   `BITE(agent=dog, patient=cat)` — the unit comprehension produces.
4. **inference** (relates propositions) — `substrate = [antecedent, consequent]`
   `IF BITE(x,y) THEN FEEL(y, BAD)` — reasoning, causation, the "condition" tower.

### Systems  (level = system — collections that serve one function)
| system | does | direction |
|---|---|---|
| `lexical-semantic` | concepts + relations (synonym, hyponym, sense) | — |
| `frames` | all predicate frames + selectional restrictions | — |
| `inference` | entailment / causation rules | — |
| `comprehension` | parse → proposition | form → meaning |
| `generation` | proposition → parse → surface | meaning → form |

### Situation  (the bound INSTANCE — what is happening now)
A situation is instance-level: propositions about **specific entities**, bound
with time, place, affect, and cause.
```
situation:s1 = { BITE(agent=e_dog, patient=e_cat) @t1 ; FEEL(e_cat, BAD) @t1 }
```
`e_dog`, `e_cat` are **entities** (instances) that instantiate the **concepts**
(types) `dog`, `cat`. Types are general and permanent; the situation is specific
and transient. Generation reads a situation/proposition and runs the frame's
selectional restrictions backward through grammar → coherent surface (no salad,
because `mouse` can't fill `agent[+pretend]` if no such frame exists).

## 2. MEMORY  (= the instance side of the same lattice)

| memory | what it is | how it's stored |
|---|---|---|
| **semantic** | knowledge: concepts, frames, inference (TYPES) | the permanent taught lattice |
| **episodic** | the situations that happened, in order (INSTANCES) | timestamped `situation` nodes |
| **entities** | the specific referents (this cat = e_cat) | `entity` instance-nodes |
| **state** | current truth (where is e_cat? how does it feel?) | a **fold** over episodic situations |
| **working** | what's salient right now (focus / recent referents) | recency+mention-ranked subset |
| **conversation** | who said what, questions, commitments | episodic situations of SAY/ASK frames |

Four design rules make this clean:

1. **State is a fold, not a store.** Current state = deterministically replay the
   episodic situations (exactly the `simulate()` we proved in the grounding
   run: replay facts → current location). So state can never desync from
   history; it's a *view*, optionally cached.

2. **Coreference is entity resolution.** "it / she / the cat" → bind to an
   existing `entity` in working memory. This is a comprehension-time step that
   turns surface mentions into instance references — and it is exactly the
   "real-text messiness" the toy reader couldn't do yet.

3. **Conversation is just episodic memory of speech-acts.** A dialogue turn is a
   situation whose frame is `SAY(speaker, content)` / `ASK` / `ANSWER`, with a
   `speaker` role. "You said X" = retrieving a remembered SAY-proposition. No new
   machinery — it falls into the same structure.

4. **The brain queries memory through the typed interface.** Resolve coreference,
   retrieve relevant propositions, fold state — all capabilities the controller
   inherits, same as every other tower.

## Why this matters for the open question
The **meaning-compounding** test is about the `frames` + `inference` systems:
do selectional restrictions and entailments, taught from one corpus, get reused
by the next (like vocabulary's 91%)? If yes, generation coherence and reasoning
both come cheap. Memory is what lets a situation persist long enough to reason
over — turning isolated comprehension into a conversation with state.

## 3. Behavior evolution — self-rewriting programs

The shift: knowledge lives in PROGRAMS (data), not frozen weights, so current
state can REWRITE it. Behavior evolves and persists; it is learned, not
prompted. (An LLM is frozen weights + ephemeral prompting — conditioning, never
learning. This is the one structural difference.)

Safety comes from the type/instance split = episodic→semantic consolidation:

- **instances rewrite constantly** (situations, entities, state) — the fold;
  cheap, safe, transient. Moment-to-moment behavior.
- **types rewrite rarely** (concepts, frames, inference) — guarded
  CONSOLIDATION: accumulated episodic evidence rewrites the shared knowledge.
  One observation updates state; a *pattern* updates knowledge.

The rewrite rule (the central unsolved mechanism) has three parts:
1. **trigger** — what evidence justifies rewriting a type (consolidation threshold)
2. **objective** — rewritten toward what (predictive success / consistency / goal)
3. **gate** — validate-before-commit + rollback, on top of schema-on-write +
   provenance, so a bad rewrite cannot corrupt the lattice.

Where this beats an LLM: continual, persistent, AUDITABLE learning from
experience — not generation.

## 4. Reversible consolidation + order dependence (how humans learn)

This is Piaget (assimilation vs accommodation) + memory reconsolidation, made
auditable. Two properties fall out — features only because knowledge is
discrete, versioned, and provenance-tracked (an LLM can have neither):

**Reversibility — consolidation is bidirectional.**
- consolidate a type on accumulated positive evidence;
- REVERT it when it produces bad outcomes enough times.
- Episodic memory keeps the originals; types carry version + provenance, so a
  single reframe can be rolled back surgically (belief revision / non-monotonic
  reasoning). An LLM cannot selectively un-learn one generalization; entangled
  weights make it impossible. Here it is a feature.

**Order materially matters — strong path dependence.**
- each example immediately rewrites the types that condition the next, so same
  inputs in a different order -> a different lattice (unlike batch-SGD LLMs).
- Feature: design the curriculum (primes -> simple frames -> complex) to
  bootstrap good foundations.
- Risk: a bad early reframe anchors everything after it (cognitive bias).
- Reversibility is the partial cure: path-dependent but self-correcting.

Hard parts (all the same learning-knot, pointed backward):
1. reversion needs an outcome signal + credit assignment (which reframe caused
   the bad outcome?).
2. entrenchment ripples: reverting a deep type invalidates dependents — but the
   substrate edges make the ripple TRACEABLE, not silent rot.
3. evaluation depends on the whole learning history -> control the order to
   reproduce behavior.

Design additions: versioned types; per-type outcome tracking; bidirectional
consolidate/revert; learning order as a first-class, auditable trajectory;
dependency-aware reversion via substrate edges.
