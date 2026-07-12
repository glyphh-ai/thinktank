# version_06_hdc_forward — HDC in the forward pass (grounded generation + refusal)

Builds on [`version_05_lattice`](../version_05_lattice) (Ada's schema *in* the
training loop) and the design doc
[`adaL/docs/glyph-ada-training-loop.md`](../../adaL/docs/glyph-ada-training-loop.md).
Where v05 put Ada's structure in the training gradient (a role head), v06 puts
the **Glyph HDC lattice in the forward pass**: it decides, as words are emitted,
whether a statement is *grounded in the distilled facts* — and refuses when it is
not.

Word-level (not char-level): the groundedness check fires at word boundaries.

## The pipeline (both sides of the lattice, one write path)

| file | what it is |
|---|---|
| `hdc.py` | vendored bipolar VSA — `bind` (⊗), `bundle` (majority-sign), `cosine = (a·b)/D`. Faithful to `glyphh-runtime` (`Encoder`, `bind(role,value)`, bipolar int8, `similarity`). |
| `corpus.py` | the joint ingestion pass: 3 domains (people / geography / chemistry) → **word+Ada-slot stream** (symbolic side) **and** per-fact **HDC cortex** (vector side), from one write path. ~53.7k tokens, 175 facts, 35 ambiguous words. |
| `model.py` | self-contained NumPy GPT (vendored from v05 `step3_gpt`) + optional **dual head** (LM + Ada role). Manual backprop, Adam. |
| `baseline.py` | the **sacred baseline** — vanilla LM. Perplexity is the number the HDC treatment must not wreck (~2.8 train / 3.2 val). |
| `memory.py` | the **groundedness primitive**: entity-keyed associative memory. `record[e] = Σ bind(role, filler)` over e's true facts; verify a claim by `unbind(record[e], role) → cleanup → filler`. |
| `forward_hdc.py` | the **decode gate**: LM proposes fillers, `probe(entity, slot)` disposes. Grounded fillers are asserted; an empty cell → geometric **refusal**. |
| `reasoning.py` | **compositional reasoning over the hierarchical lattice**: encode each record as a tree (`cortex → layer → role → filler`) and navigate it — subtree extraction, bounded-depth multi-hop, relational analogy. |
| `lattice_adapter.py` | **Ada as an in-loop memory adapter on a stock model**: a logits processor that masks the vocabulary, every step, to Ada's grounded fillers (from the HDC `probe`) or a refusal token. Model-agnostic, MLX-optional; CPU-unit-tested. |
| `run_qwen.py` | drives a **real stock Qwen3** (MLX, Apple Silicon) with the adapter — grounded answer or refusal, regardless of pretraining. |
| `business.py` | an **organization on the lattice**: exact store (revenue, cases…) for exact aggregation + an HDC semantic layer for **similarity cohorts** ("customers likely to churn"). Deterministic query engine, self-verified. |
| `run_qwen_analytics.py` | **consistency test**: analytical queries answered by stock Qwen3, unconstrained (sampled) vs adapter-constrained. |
| `plan.py` | **typed query-plan schema + executor**: NL → a plan confined to real ops/fields/values → deterministic execution over the lattice. Self-verified. |
| `run_qwen_planner.py` | **NL → constrained plan → answer**: embedding router does semantic routing (metric/churn/refuse), the LLM fills the structured slots (constrained), lattice executes + constrains the answer. |
| `schema_guard.py` | **semantic schema-membership**: nearest-exemplar embedding router (`Qwen3-Embedding-0.6B`) with an out-of-schema negative class → refuse off-schema questions + route the metric. |
| `adal_sdk.py` | **the bolt-in SDK facade**: `ingest(records)` → both structures; `probe/verify/aggregate`; `constraint(...)` → a logits processor any logit-access LLM plugs into its decode loop. |

## The key design correction

Groundedness is **not** whole-sentence cosine (bundle a sentence's role-fillers,
cosine-match to stored sentence cortices). That is surface matching — it flails in
a mushy ~0.5 band and never cleanly locks. The right primitive is **entity-keyed
retrieval**: keep the subject as the *retrieval key* and probe each slot by
unbind+cleanup. Then:

- **single-fact membership** is local, per-word, and clean
  (true bindings mean cos **+0.30** vs random **+0.00**);
- **multi-hop entailment** is the genuinely compositional hard edge — chained
  `unbind → cleanup → re-probe`, with SNR decay per hop.

## Results

```
grounded decode:  erin lives in -> boston (+0.219)   ASSERTED
                  helium is a   -> gas    (+0.274)   ASSERTED
                  erin is from  -> maine  (+0.221)   ASSERTED
refusal (record's location fact deleted):
                  erin lives in -> REFUSE (best is 'tucson' +0.021, empty cell)
```

Run any file with `python <file>.py` (each self-verifies).

## Compositional reasoning over the hierarchy (`reasoning.py`)

Encoding each record as a tree (nested binds: `cortex → layer → role → filler`)
gives three capabilities a flat bundle cannot, plus one honest non-win:

- **subtree extraction** — `unbind(cortex, layer_key)` reads a whole layer at once
  (`erin spatial → {location=boston, origin=maine}`). Flat memory can't address a layer.
- **bounded-depth multi-hop** — retrieve → **cleanup to a clean concept** → retrieve.
  A 6-hop chain holds margin **~0.37 at every hop** (no decay): depth is bounded
  only by per-hop SNR, and the hierarchy's tiers are the cleanup boundaries.
- **relational analogy** — the role is *discovered*, not named: "maine is to erin
  as ? is to jack" → discovers `spatial.origin`, returns `kansas`. (Cleanup of the
  extracted role is essential — the raw extraction leaks the source filler back.)
- **capacity (honest)** — within capacity, a single flat bundle *beats*
  double-sign-bundled hierarchy on raw leaf SNR. Hierarchy's payoff is **structure**
  (addressing + cleanup boundaries for depth), not leaf SNR.

**Takeaway:** deeper reasoning = `retrieve → cleanup → retrieve`; the hierarchy
supplies the natural cleanup points, so chains stay bounded-error to arbitrary depth.

## In-loop memory adapter on a stock model (`lattice_adapter.py`, `run_qwen.py`)

The guarantee is **architectural, not learned**: the base model's pretraining is
irrelevant to Ada-groundedness. A logits processor masks generation, at every step,
to the token continuations of Ada's grounded fillers for `(entity, role)` — or a
refusal. By construction the model cannot emit a non-Ada answer. Demonstrated on
**stock Qwen3-1.7B (MLX)**:

```
Erin lives in the city of…   Qwen: "New York, and she is 1"   ->  Ada-grounded: boston
Jack works as a…             Qwen: "carpenter. He has 10"     ->  Ada-grounded: farmer
[delete erin's location]     Qwen: "New York..."             ->  Ada-grounded: unknown (refuses)
```

The guarantee is size-independent (size only affects fluency), so 7B/14B is a
one-line `MODEL=` change. Scope: it adjudicates **Ada-typed facts** — statements
that decompose to `(entity, role, filler)`. Knowing *which* slot is being answered
is currently supplied (structured QA); free-form detection of "the model is now
asserting fact X about entity Y" needs an inline role/entity tagger (the v05 role
head) — that's the remaining seam, not a hole in the guarantee.

Run:  `/opt/homebrew/anaconda3/bin/python run_qwen.py`  (needs `mlx_lm`, Apple Silicon)

## Analytical queries & consistency (`business.py`, `run_qwen_analytics.py`)

Can a stock LLM answer complex questions (total revenue, churn risk) consistently
when wired in-loop to the lattice? Yes — but the honest architecture matters:

- **exact aggregation → deterministic reducer over the exact store.** HDC does NOT
  do the arithmetic (vectors are lossy at numbers; that would be *inconsistent*).
- **semantic cohort ("likely to churn") → HDC cosine to a prototype** built from
  known-churned records. This is the part SQL can't express; the cohort came back
  6/6 declining/dormant with no rule written.
- **consistency → the adapter constrains the model to the computed answer.** The
  model gets fluency; the lattice keeps authority over the number.

Measured on stock Qwen3-1.7B, K=5 samples/query:

```
                     unconstrained (sampled)        adapter-constrained
total revenue        4 distinct, 0/5 correct   ->   2054653  (5/5, identical)
enterprise-west rev  3 distinct, 0/5 correct   ->   265823   (5/5)
most likely to churn 5 distinct, 0/5 correct   ->   acme14   (5/5)
```

The model can't sum 54 numbers, filter-then-sum, or rank churn — alone it's wrong
*and* non-reproducible. Consistency doesn't come from the model reasoning better;
it comes from **removing the model's authority over the answer.**

## NL → constrained query-plan (`plan.py`, `run_qwen_planner.py`)

Closes the last seam: the model turns a question into a **typed plan**, but each
plan slot (`op`, `field`, `tier`, `region`) is constrained to its schema enum —
so the model does the language mapping yet **cannot emit an invalid or ungrounded
plan**. The plan executes deterministically; the answer is constrained to the
result. Consistency holds at *both* layers. On stock Qwen3-1.7B (K=3 runs each):

```
"total revenue, all customers"          -> total(revenue)                     2054653  ✓
"revenue from enterprise, west coast"   -> total(revenue tier=ent region=west)  265823  ✓
"how many pro-tier customers"           -> count(tier=pro)                          19  ✓
"which customer most likely to churn"   -> churn()                             acme14  ✓
"average revenue per enterprise cust."  -> average(revenue tier=ent)            168005  ✓
"total carbon footprint"                -> total(revenue)  [MISS: should refuse] 2054653
   → all 6: 1 distinct plan + 1 distinct answer across 3 runs (perfectly consistent)
```

**The fix — put each decision on the right substrate.** The 1.7B's `op`/refusal
classifier was unstable (whack-a-mole). So we split the plan by what each part
actually needs:

- **semantic routing** (metric? churn? out-of-schema?) → an **embedding router**
  (`schema_guard.py`, `Qwen3-Embedding-0.6B`), nearest-exemplar with an explicit
  out-of-schema negative class. Robust and **size-independent**: 8/8 on the guard
  test, clean margin (in-schema +0.17..+0.30 vs out-of-schema all negative).
- **structured extraction** (aggregation + tier/region) → the LLM, each slot
  constrained to its enum.
- **execute + answer** → deterministic lattice + answer constraint.

With the router doing the semantic decisions, the full pipeline goes **7/7 correct
and 7/7 consistent** on stock Qwen3-1.7B — including churn routing and out-of-schema
refusal, the two the LLM alone kept missing.

**Important correction:** this membership check uses **semantic embeddings, not our
random-atom HDC.** HDC vectors are quasi-orthogonal by construction — perfect for
binding/structure, but they carry *no semantic content* ("revenue" ⟂ "earnings" ⟂
"carbon"), so an HDC cosine cannot tell that "total earnings" is the revenue field
and "carbon footprint" is not. Semantic membership needs semantic vectors. (The two
layers compose: embeddings for *what the question means*, HDC for *how the fact is
structured/bound*.)

Residual axis: the LLM's structured extraction still scales with model size (a
mis-extracted filter is a consistent, grounded aggregate — never a fabricated
number). That's the 7B/14B lift.

## Open / next

1. **Bigger model on structured extraction** — the remaining error axis; run
   `run_qwen_planner.py` with `MODEL=` a 7B/14B (Modal/Colab) and measure the
   filter/aggregation accuracy lift (semantic routing is already size-independent).
2. **Enricher label-noise** — records here are distilled from clean corpus
   fields; the real test is Ada's enricher output (a few % noise).
3. **Free-form fact-assertion detection** — the v05 role head as a second logits
   stream, so the gate fires on open generation, not just templated slots.
4. **Trained-in binding** — currently HDC is only the runtime *disposal* layer;
   the doc's other layer (bind HDC targets into the weights) is not yet wired in.
