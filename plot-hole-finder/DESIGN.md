# plot-hole-finder — Design

Detect plot holes, broken references, and inconsistencies in very large
documents (200k+ words ≈ 270k+ tokens), well beyond a single context window.

## The core insight

This is **not a retrieval problem, it's a contradiction-detection problem.**

Pure vector RAG answers "find passages similar to query X." But:

- An inconsistency has no query. You must enumerate *all* assertions and check
  them, not fetch top-k for a question.
- Contradictions are usually semantically **similar** ("her eyes were blue" vs
  "her eyes were brown" embed close together), so similarity ranking actively
  works against you — it can't tell agreement from contradiction.
- The two halves of a contradiction can sit 150k words apart. You need
  exhaustive coverage of a *slot*, not nearest neighbors.

So RAG is demoted from the *detection* layer to the *evidence* layer: once we
suspect a contradiction, RAG pulls the actual passages so an LLM can confirm it
and quote chapter-and-verse (kills hallucinated "plot holes").

## The key reframe: compress prose → world model, then reason

The prose is 270k tokens. The **structured facts extracted from it** (entities,
attribute assertions, events, references) are maybe 20–50k tokens. That smaller
world-model *does* fit in context with room to reason — so we spend the LLM's
attention on the compressed knowledge representation, never on raw prose at
scale. Big context windows help, but they don't fix recall of subtle
cross-references buried far apart, and re-running full passes on every edit is
expensive. Extraction-then-reasoning is about **reliability and cost**, not just
fitting.

## Pipeline (a triage funnel)

Expensive reasoning is rationed; cheap signals do the wide scan first.

```
ingest → chunk → EXTRACT (LLM map, parallel) → index (3 stores)
       → CANDIDATE GENERATION (cheap: rules + HDC triage)
       → VERIFY (expensive: LLM + RAG evidence)  → report
```

### 1. Ingest & chunk

Structural chunking by scene/chapter (not fixed-size), preserving narrative
order and char offsets. Every fact stays traceable to a passage.

### 2. Extract (map step — parallel, bounded per chunk)

Per chunk, an LLM emits a normalized schema:

- **Entity** — id, type (character/object/location/concept), canonical name,
  aliases.
- **Attribute assertion** — entity, attribute, value, polarity, narrative_time,
  source offset, confidence. ("Elena | eye_color | brown @ ch.20")
- **Event** — participants, action, location, narrative_time. Crucially includes
  **transformation events** (dyes hair, object destroyed, character dies).
- **Relationship** — subj, rel, obj, time.
- **Reference** — mention + type: *setup* vs *payoff/callback*. Drives
  broken-reference detection (payoff with no setup, setup never paid off).
- **Knowledge state** — "entity knows fact F as of time T" (for "character knows
  something they couldn't").

### 3. Three stores, three jobs

- **Structured fact store** (Postgres) — the canonical entity/state/event
  timeline. The "world model." Grouping by `(entity, attribute)` is what makes
  consistency checks tractable.
- **Vector store** (pgvector, same DB) — *evidence retrieval only*.
- **HDC memory** (VSA, e.g. `torchhd`) — cheap, massively parallel triage +
  entity resolution at scale (see below).

### 4. Candidate generation (cheap, wide)

The naive approach is O(n²) over tens of thousands of facts — infeasible. We
narrow it three ways:

- **Slot bucketing:** only compare assertions about the same `(entity,
  attribute)`. A contradiction is only real if states differ *and no licensing
  transformation event sits between them* — i.e. blue@t1 → brown@t2 is fine if
  "dyed hair" happens in (t1,t2); flagged if not. This "find the event that
  licenses the change" framing is the heart of continuity checking.
- **Reference graph:** every payoff must trace to a prior setup; flag dangling
  setups and unsupported payoffs.
- **Temporal/causal:** build the event timeline; flag ordering violations and
  knowledge a character can't yet have.

### 5. HDC's actual job

HDC isn't the reasoner — it's a cheap recall/triage substrate that shines
exactly where the structured store is weakest:

- **Coreference at scale:** "the captain" / "she" / "Elena" / "Mrs. Vance" =
  one entity across 200k words. Bundle mention contexts into hypervectors and
  cluster — embarrassingly parallel, O(D) ops. Without identity resolution you
  can't detect that "her" age contradicts.
- **State accumulation & incompatibility signal:** an entity's hypervector
  accumulates bound `(attribute ⊗ value)` pairs. A new `(attribute ⊗ value')`
  binds to the same attribute role but unbinds to a filler that won't clean up
  to the stored value → a cheap "something's off here" flag, even for facts that
  never got cleanly typed by extraction.
- **Fuzzy fallback:** catches near-miss contradictions the rigid schema misses.

HDC narrows the candidate space; the LLM verifier provides precision and
explanation. Good division of labor.

### 6. Verify (expensive, narrow)

Each candidate → LLM verifier with RAG-retrieved evidence passages: confirm or
reject, assign severity, explain, quote both passages. Bounded count because
triage already filtered.

### 7. Report

Severity-ranked issues, each with both passages, jump-to-offset, and a category
(continuity / broken reference / timeline / character / knowledge).

## Hard problems we'll wrestle with

- **Legitimate change vs error** — the licensing-event check above. Biggest
  false-positive source.
- **Narrative time vs story time** — flashbacks mean a *later* passage narrates
  an *earlier* in-world event. We need both orderings; the "character knows X"
  check depends on story-time, not text order.
- **Coreference precision** — over- or under-merging entities poisons everything
  downstream.
- **Capacity/crosstalk** in HDC superposition — needs cleanup memory; tune
  bundling limits.

## Stack

- **Server:** Python, FastAPI. Async job pipeline (extraction is the long pole)
  via a task queue (arq/Celery + Redis) or DB-backed tasks for the demo.
  Progress streamed to the client (SSE/WebSocket).
- **LLM:** Claude (Anthropic API) for extraction + verification.
- **Data:** Postgres + pgvector (structured facts *and* vectors in one DB).
  HDC vectors via `torchhd`, persisted alongside.
- **Front end:** Next.js (App Router) — upload, live progress, results browser
  with evidence highlighting and jump-to-passage.

## Open questions

- Build order: full vertical slice first, or core engine (extract → check)
  before any UI?
- HDC in v1, or stand up structured + RAG baseline first and layer HDC on once
  there's a working pipeline to measure it against?
- Confirm Claude as the LLM.
