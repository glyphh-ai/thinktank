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
