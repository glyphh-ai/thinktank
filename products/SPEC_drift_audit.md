# SPEC — Glyphh Drift: semantic changelogs for fine-tuning

**Handoff document for a coding agent.** The working prototype exists in
this repo at `demo_model_diff/` and produced a real finding on its first
run; this spec turns it into a CLI product. Read `demo_model_diff/diff.py`,
`version_12_wide_seed/wideband.py`, and `paper/README.md` first.

## 0. What it is

A CLI (`drift`) that answers, with named numbers, the question every
fine-tuning team asks and none can currently answer: **"what did this
checkpoint change about meaning?"** Point it at two model checkpoints
(base vs fine-tune, week N vs N+1, yours vs vendor update); it emits a
**semantic changelog**: which named axes moved, which words/concepts moved
most, discrete committed-bit flips, and per-axis stability — plus alerts
when watched axes (safe–dangerous, honest–dishonest, …) cross thresholds.

Proven output from the prototype (Qwen2.5-0.5B base→Instruct): median axis
stability 0.993, zero bit flips, top mover "mean" (statistics → unkindness,
Δ toward bad/rude/angry/false/dishonest). A quiet, correct changelog with a
real catch — exactly the product behavior.

## 1. Why it wins

Everything else is behavioral (benchmarks, eval suites — coarse aggregates
over prompts someone thought to write) or heavyweight interp
(SAE/crosscoder diffing — GPU-days, unnamed features). Drift is: forward
passes only, minutes per checkpoint pair, every line named and
human-readable, deterministic, CI-able. It reads *representations*, so it
catches drift no output eval samples.

## 2. CLI UX (v0 deliverable — REPL/CLI only)

```
drift snapshot Qwen/Qwen2.5-7B --out base.snap          # extract once, reuse
drift snapshot ./checkpoints/step-4000 --out ft.snap
drift diff base.snap ft.snap --report changelog.md      # the changelog
drift diff base.snap ft.snap --watch safe-dangerous,honest-dishonest \
      --fail-on "watched: |dz|>0.5 for >20 words"       # CI mode: exit code
drift timeline *.snap --axis good-bad                   # drift over N ckpts
drift repl base.snap ft.snap                            # interactive: query
  > word mean          # per-axis movement of one word
  > axis polite-rude   # top movers on one axis
  > flips              # committed bit flips
```

Snapshots are the unit: one extraction per checkpoint (~16k forward
passes; minutes on GPU, ≤1 h CPU for 0.5B-class), diffs are instant dot
products. A snapshot is portable and tiny relative to the model (vectors
for the vocab at selected layers, ~100–400 MB; offer `--layers 12` and
fp16 to shrink).

## 3. Method (port, don't reinvent — `demo_model_diff/diff.py` is the reference)

1. Fixed vocabulary (ships with the tool: the 1375-word diff list; user-
   extensible with `--extra-words domain_terms.txt` — domain words are
   where fine-tune drift actually lives, make this loud in docs).
2. 12-frame extraction per word per checkpoint (protocol from
   `version_07_prime_basis/step1_vectors.py`), residual states, layer
   auto-selected per model family by the certification recipe (split-half
   reliability for measurement + a small human-norm check for depth — see
   v11 FINDINGS "reliability is not validity").
3. **Own-frame axes**: rebuild the 242 named axes from each checkpoint's
   own seed vectors; z-score readings within checkpoint; diff the z's.
   This is the load-bearing methodological trick — never compare raw
   coordinates across models.
4. Report: axis stability ranking, top movers (global + per watched axis),
   committed bit flips (|z|≥1 both sides, sign reversed), markdown +
   JSON output.

## 4. Product features beyond the prototype

- **CI mode**: thresholds + nonzero exit + JSON for pipelines; GitHub
  Action wrapper.
- **Timeline mode**: N snapshots → per-axis drift curves (catch slow drift
  that pairwise diffs miss).
- **Domain packs**: user vocabularies + user axis pairs (their jargon's
  contrasts), auto-certified with the v10 dead-zone recipe before use —
  refuse uncertified axes with a clear message (the certification IS the
  brand).
- **Multi-arch support**: anything exposing hidden states via HF
  transformers. Validate per family with the v11-style gate suite
  (automated, half a day per family); maintain a supported-families table
  with honesty about what's validated vs best-effort.
- **Scale note**: 7B/70B are fine (forward passes, short sequences —
  batch well). Closed API models are out of scope by physics (no hidden
  states); say so in docs rather than faking it with output-only proxies.

## 5. Build phases

| phase | deliverable | gate |
|---|---|---|
| 1 | `snapshot` + `diff` + markdown report | reproduces `demo_model_diff/REPORT.md` bit-for-bit on the 0.5B pair |
| 2 | watch/fail-on + JSON + GH Action | a seeded synthetic drift (LoRA-tune a toy model on skewed data) is caught; clean pair passes |
| 3 | repl + timeline | — |
| 4 | domain packs + auto-certification | uncertified axes refused; certification report emitted |
| 5 | 7B validation run (Qwen2.5-7B base/Instruct) | axes certify; changelog qualitatively richer; publish as the flagship example |

## 6. Tech

Python 3.11, torch (CUDA optional, CPU works), transformers, numpy, scipy,
click/typer for CLI, rich for terminal output. No services, no DB — files
in, files out. Pin and hash the axis bank and calibration tables; print
their versions in every report header (determinism/auditability claim).

## 7. Risks

- Word-level drift can be quiet while behavior changes a lot (our own 0.5B
  finding). Mitigate in docs (this measures representational drift) and in
  roadmap: a probe-prompt generation-metering mode (meter the *outputs* of
  both checkpoints on a fixed prompt suite) is the v2 feature that closes
  the behavior gap — spec it only after v0 ships.
- Cross-family comparisons (Llama vs Qwen) are NOT supported in v0 —
  own-frame z-scores make within-family diffs valid; cross-family needs
  care. Refuse with a clear error.
