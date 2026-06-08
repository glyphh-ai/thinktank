# Evaluation harness

A **comparative** benchmark: the same labeled corpus is run through every
detection approach and scored head-to-head. This is how we decide the
architecture with numbers instead of assertions.

```bash
ANTHROPIC_API_KEY=... python -m eval.harness
```

Real Claude end to end (no fallbacks). Embeddings here are evidence-only and not
scored, so the harness uses a tiny deterministic embedder to avoid a model
download; production uses FastEmbed.

## Corpus (`corpus.py`)

A generated ~18k-word document seeded with **~28 planted inconsistencies** across
varied types (eye colour, age, role, height, hometown/knowledge, object status,
sibling count) over many distinct entities, plus **~9 decoys** — changes an
in-story event legitimately explains (a dyed haircut, a promotion, a house move)
that must **not** be flagged. Deterministic (seeded). It's a controlled answer
key — the honest way to score when no public ground truth exists. Size it up
(`build_corpus(num_plants=..., target_words=...)`) for more statistical power.

## Arms

| Arm | What it does |
| --- | --- |
| `claude_1m`  | Hand the whole document to Claude in one pass ("find all inconsistencies"). The baseline. |
| `structured` | extract → glyphh cortex per-(entity,attribute) candidates → filter. |
| `holistic`   | extract → one LLM pass over the compressed world model → filter. |
| `hybrid`     | **recall-first**: union of recall arms (single-pass + structured + holistic) → one **precision filter** that drops event-licensed changes. The product. |

## Scoring (`scoring.py`)

- **recall** = planted inconsistencies found / total plants
- **precision** = issues that match a plant / total issues raised
- **decoyX** = decoys wrongly flagged (licensed changes) — a precision failure
- Matching is by entity + attribute *concept* (synonyms), not exact strings, so
  "role" matches an issue that says "profession".

Also reports **cost** (real token usage × model pricing) and **latency** per arm.

## Results snapshot (Opus 4.8, 16.6k-word corpus, 28 plants + 9 decoys)

```
arm           recall   prec    f1  issues  decoyX   FP   cost$   sec
claude_1m      100%   100%  100%     28       0     0   0.223    25
structured      68%   100%   81%     19       0     0   0.318    52
holistic        75%    88%   81%     24       3     3   0.414    82
hybrid         100%   100%  100%     29       0     0   0.646   113
```

**Reading it honestly:** on a document that *fits in context*, the plain
single-pass baseline ties the hybrid at 100/100 for ~1/3 the cost and ~1/4 the
latency — the pipeline adds no quality there.

## Beyond the context window (`python -m eval.beyond_context`)

The decisive test: a causal plot hole whose halves are **>1M tokens apart** (a
vault sealed forever / key melted / lock welded, opened ~1M tokens later).

```
document: 887,719 words = 1,681,027 tokens (OVER the 1M context window)
[baseline] single pass: FAILED — 400 "prompt is too long: 1681330 > 1000000 maximum"
[pipeline] 23 chunks -> 9-fact world model -> holistic review:
           plot hole detected: YES (1/1) | 88s | $8.50
```

The baseline **cannot run** past 1M tokens; the pipeline chunk-extracts, compresses
to a tiny world model, and the holistic pass catches the violation across ~1.66M
tokens.

## Cross-chunk architecture bake-off (`python -m eval.crosschunk`)

The in-context corpus sliced into small chunks so every plot hole spans chunk
boundaries — comparing approaches that all read in chunks:

```
arm                 recall   prec   decoyX   cost$
chunked_baseline      61%    63%       7    0.23   (no memory across chunks)
rolling_memory       100%   100%       0    0.39   (Claude maintains memory.md)
pipeline              50%   100%       0    0.41   (extract -> world model -> review)
```

**Rolling memory (the Agent-SDK memory-tool pattern) wins** — best recall, perfect
precision, and the simplest design. The bespoke extraction+cortex pipeline came
last on recall: the rigid extraction schema drops facts, so the world model is
lossy. Letting Claude keep a running memory beats hand-built extraction.

## Verdict

| Regime | Best approach |
| --- | --- |
| Fits in context (≤1M tokens) | **single Claude pass** — ties the pipeline, cheaper, faster |
| Beyond context (>1M tokens)  | **the pipeline** — the baseline cannot run; pipeline detects (1/1) |

The pipeline earns its existence exactly where the project began: *well beyond
the context window.* The product should route adaptively — single-pass when it
fits, chunk-and-reason when it doesn't.

