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

## Open / next

1. **Enricher label-noise** — records here are distilled from clean corpus
   fields; the real test is Ada's enricher output (a few % noise).
2. **Trained-in binding** — currently HDC is only the runtime *disposal* layer;
   the doc's other layer (bind HDC targets into the weights, per-node deep
   supervision) is not yet wired in.
3. **Deeper trees** — records are cortex→layer→role (3 tiers); Ada's full schema
   adds segments, and per-hop fan-out sets the SNR bar that limits chain depth.
