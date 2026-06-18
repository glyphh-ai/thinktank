# Handoff — adaL next step: a local fine-tuned enricher (and how to benchmark it)

**Audience:** an agent working in the `glyphh-ai/adaL` repo.
**Origin:** experiments in `timmetim/thinktank` (`frontier/`, `version_05_lattice/`).
This is self-contained; read it before writing code.

---

## TL;DR decision (don't re-chase dead ends)

**Build a standalone, locally-runnable model that does `messy text -> Ada universal
schema` — a drop-in replacement for the per-fact frontier-API enricher — then
benchmark it hard against that enricher (the teacher/ceiling) and end-to-end on
Ada's existing query benchmark.**

Do **NOT** put the schema into a *generator's* training loss ("in the loop").
We tested that thoroughly; see "What we learned" #1. The value is a **decoupled
extractor**, not a schema-aware generator.

---

## What we learned (load-bearing, from the thinktank experiments)

1. **In-loop schema (schema as a loss target inside a generator) is a dead end
   for capability.** It's learnable, free (no LM-loss cost), and scales (99.7%
   role-tag accuracy on Qwen3-8B) — but it bought **no downstream capability**.
   The one clean test (confidence calibration) was **null**: a plain LM already
   knows what it didn't memorize, so structure added nothing. Everything useful
   reduces to (a) the store and (b) extraction — neither needs the generator's
   loss touched. **Conclusion: decouple. Extractor only.**

2. **The real win is the store (Ada) + cheap extraction.** Architectural scaling
   harness (`frontier/scale_bench.py`), exact-index vs RAG vs bare context, 1M ->
   **1B** facts: exact substrate answers recall/aggregation/refusal/belief at
   100% in seconds; RAG's dense index is physically unbuildable past ~10M (1 TB
   at 1B); bare context never competes. This is Ada's existing thesis, confirmed
   to 1B. **Ada the store is the asset.**

3. **Prompting alone does NOT reproduce Ada's schema.** Zero-shot Qwen3-8B on
   messy Wikipedia prose scored **slot-F1 0.12** vs the enricher's gold. So you
   cannot skip training — a *fine-tuned* extractor is required. This is the gap
   to fill.

4. **Messy text is genuinely harder than the clean case** because slot values are
   paraphrased/implicit ("the Texas capital" -> `spatial.location=austin`), so
   the task is **generation/extraction, not span-tagging.** The student must
   generate the schema, not tag tokens.

---

## The task

Fine-tune Qwen3 (start with the size you can serve; sweep down for the runtime)
to map text -> Ada's 7-layer/33-role universal schema. One model, two axes:

- **(a) schema-on-write fidelity:** emits valid JSON, only legal roles, leaves
  absent slots empty (no hallucinated slots). Precision side.
- **(b) messy-text robustness:** does (a) from real messy prose, across domains.
  Recall/generalization side.

**Gold = `ada/cognitive/universal.py::UniversalEnricher` output** (the existing
frontier-API enricher). It is the teacher and the ceiling. The whole point is to
match it **locally and cheaply** (no per-fact API call).

---

## Benchmark it hard (pre-register every bar before running)

Reuse Ada's protocols: `benchmark/PROTOCOL.md`, `benchmark/phase1` (extraction,
the teacher hit 99.8% semantic / 3.7% spurious-fill), `benchmark/phase3` (NL
query benchmark, 1K->1M facts).

| dimension | metric | notes |
|---|---|---|
| schema fidelity | % valid JSON, % legal roles, **spurious-fill rate** | target spurious-fill < 3.7% (Ada Phase-1 teacher) |
| extraction vs teacher | slot-F1, P/R — split **clean / paraphrase / implicit** | teacher = 1.0 ceiling |
| domain transfer | train chem+law, test philosophy/general | memorize vs generalize |
| refusal / ∅ | absent-slot precision (don't invent slots) | Ada's honesty property |
| cost / speed / size | docs/sec (CPU+GPU), $/1M facts, MB — **vs API enricher** | the reason to do this at all |
| size sweep | F1 vs size vs cost: 8B / 4B / 1.7B / 0.6B | find the smallest viable runtime |
| **end-to-end (killer test)** | ingest a corpus via the fine-tuned extractor, run **phase3** | does NL-query accuracy hold vs enricher-ingested? |

**Acceptance:** the smallest model whose **end-to-end phase3 accuracy ≈
enricher-ingested** (within a few points) at a **fraction of the cost** and **no
API dependency**. That's the deliverable: a local enricher Ada can ship.

---

## Artifacts to copy/adapt from `timmetim/thinktank/frontier`

- `messy/fetch.py` — pull real messy Wikipedia prose (`--random` for volume).
- `messy/enrich.py` — call the REAL `UniversalEnricher` to make gold `(text,slots)`.
- `messy/eval_messy.py` — `slot_f1` metric (robust to dict/list/scalar layers) +
  zero-shot baseline. **Reuse `slot_f1` as the core scorer.**
- `messy/finetune_messy.py` — LoRA fine-tune Qwen3 to generate slots from text
  (text->JSON). This is the seed of the real extractor; harden it.
- `prepare_data.py` — render adaL's existing enriched facts to (text, slots).
- `scale_bench.py` — the store-scaling benchmark (context for why the store wins).

These run; CPU-validatable parts (fetch, slot_f1) are validated. The Qwen3 runs
need a GPU. `enrich.py` needs `ANTHROPIC_API_KEY`.

---

## Honest open risks

- **Distillation may not match the teacher on the long tail** (rare domains,
  exotic entities). Measure per-domain; expect a quality/size tradeoff curve.
- **Spurious fills** (hallucinated slots) are the dangerous failure for a
  schema-on-write system — weight precision, not just F1.
- **Don't trust tiny eval sets.** A 67-pair smoke run is meaningless; use
  thousands of held-out `(text, slots)` pairs with a clean train/test split
  (no entity/domain leakage).
- **The teacher is itself imperfect** (~99.8%, 3.7% spurious). The student's
  ceiling is the teacher; if you need better than the teacher, that's a separate
  problem (improve the enricher prompt/gold first).

---

## One-line summary for the PR

> Replace Ada's per-fact frontier-API enricher with a fine-tuned local model
> (`text -> universal schema`), benchmarked to match the enricher's extraction
> and preserve downstream query accuracy at a fraction of the cost — proven by
> end-to-end phase3, not just slot-F1.
