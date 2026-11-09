# FINDINGS — version_22_qwen_adapter

A counter memory beside Qwen3-1.7B-4bit (MLX). 300 invented facts ("Piheba
Doti keeps a hawk named Nipira."), each told twice, then a third corrected
twice. Right name must beat 20 others. English damage = Qwen bits/char on
20k chars of Pride and Prejudice (1.164 untouched).

| arm | recall | reworded | corrected (reworded) | untouched | English |
|---|---|---|---|---|---|
| Qwen alone | 5% | 6% | 1% (4%) | 4% | 1.164 |
| + exact memory (token-hash counters) | 99% | 6% | 100% (4%) | 100% | 1.164 |
| in context (every fact in the prompt) | 99% | 92% | 6% (17%) | 100% | n/a |
| LoRA, best of 1e-5 / 1e-4 / 1e-3 | 7% | 6% | 35% (25%) | 7% | 1.29–5.14 |
| **+ glyphh memory (step3)** | **100%** | **100%** | **99% (95%)** | **100%** | **1.172** |

- **Exact memory** learns once, takes corrections and forgets nothing, but
  only for the wording it was taught.
- **In context** reads rewordings but can't take a correction at 300 facts:
  the old name keeps 94%.
- **LoRA** forgets untouched facts and damages English at every rate.
- **Glyphh memory** (step3_glyphh.py): the same counters, keyed on a glyphh of
  the last 8 content words (Ada's lexicon word glyphs + NSM primes, unknown
  words such as invented names as their own symbols), nearest key above tau
  (0.406, set on dev facts against unrelated text). It is the only arm that
  wins on every column.

Caveats: one rewording template; every person owns one animal, so the
invented names may carry the match alone; 300 facts. step4 tests those.

Logs: step1_arms1-4.log, step1_context_lora.log, step3.log
(step1_results.json is the earlier 30-fact run).
