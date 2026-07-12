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

## Open / next

1. **Deeper compositional reasoning over the hierarchical lattice** — multi-hop
   entailment is the truly hard part; the lattice is hierarchy-based, so probe
   composition should exploit that structure.
2. **Enricher label-noise** — records here are distilled from clean corpus
   fields; the real test is Ada's enricher output (a few % noise).
3. **Trained-in binding** — currently HDC is only the runtime *disposal* layer;
   the doc's other layer (bind HDC targets into the weights, per-node deep
   supervision) is not yet wired in.
