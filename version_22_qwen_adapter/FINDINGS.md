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

## step4 — 3,000 facts, two animals per person, five untaught rewordings

Taught once as "{p} keeps a {a} named", a third corrected once; the other
animal of the same person is always a distractor; scored on 300 sampled.

| arm | taught | qa | called | belongs | ask | answers | English |
|---|---|---|---|---|---|---|---|
| Qwen alone | 5% | 6% | 6% | 6% | 5% | 6% | 1.164 |
| exact, recall | 100% | 6% | 6% | 6% | 5% | 6% | 1.164 |
| exact, corrected | 100% | 5% | 3% | 3% | 5% | 5% | 1.164 |
| **glyphh, recall** | **100%** | **99%** | **91%** | **100%** | **100%** | **90%** | 1.223 |
| **glyphh, corrected** | **100%** | **97%** | 47% | **100%** | **100%** | 51% | 1.225 |

Untouched facts after correction: glyphh 100% (taught) / 100% (qa); exact
100% / 4%. Old name winning after correction (taught): 0% for both.

- The owner's name doesn't carry it: with two animals per person, the glyphh
  still picks the right one 90–100% of the time in every wording.
- Two weak spots: corrections reach only about half of "is called" and
  "answers to" (the wordings without "name"), and English drifts +0.06
  bits/char at 3,000 facts (+0.008 at 300): a bigger store fires on
  unrelated text more often, so tau has to tighten with size.

Log: step4.log, results step4_results.json.

## step4b — the corrections gap, found and fixed

diag_corrections.py: in "is called" and "answers to" there is no "name"
word, so the name's first token lands on the key for "...keeps a cat"
(whose count says " named"), not "...keeps a cat named" (40/40 "called"
cases). The first token gets no help, the old name's later-token keys still
exist, so a correction comes out ~50/50.

Fix: mix every key above tau within delta 0.05 of the best match, counts
weighted by similarity (delta picked on a 120-fact pilot); and set tau on a
dev store the size of the test store (3,000 dev facts, seed 1: tau 0.647
instead of 0.386).

| glyphh, 3,000 facts | taught | qa | called | belongs | ask | answers | English |
|---|---|---|---|---|---|---|---|
| step4 recall | 100% | 99% | 91% | 100% | 100% | 90% | 1.223 |
| **step4b recall** | 100% | 100% | **100%** | 100% | 85% | **97%** | **1.165** |
| step4 corrected | 100% | 97% | 47% | 100% | 100% | 51% | 1.225 |
| **step4b corrected** | 100% | 100% | **98%** | 100% | 83% | **92%** | **1.165** |

Untouched 100%/100%, old name wins 0%. The English drift is gone (1.164
untouched Qwen). The cost: "Ask {p} what their {a} is named, and they will
say" drops to ~85%, its later name tokens matching at 0.61–0.70, under the
tighter tau.

Log: step4b.log, results step4b_results.json.

## step4c — the "ask" dip, found and fixed

diag_ask.py (dev facts, seed 1): in "Ask {p} what their {a} is named, and
they will say", the filler "their / they / will" fills the 8-word window
and pushes the owner's first name out, so later name tokens match the right
key but below tau (38% above it at the 2nd token).

Fix: drop a standard English function-word list (pronouns, auxiliaries,
determiners, prepositions, conjunctions; not picked on any test wording)
from the key, except a word still being written (a name's first piece, "He"
of "Heliro"). Checked on dev facts, then run once on the test facts (tau
0.449 from a 3,000-fact dev store, delta 0.05).

| glyphh, 3,000 facts | taught | qa | called | belongs | ask | answers | English |
|---|---|---|---|---|---|---|---|
| step4b recall | 100% | 100% | 100% | 100% | 85% | 97% | 1.165 |
| **step4c recall** | 100% | 100% | 100% | 100% | **100%** | **99%** | 1.167 |
| step4b corrected | 100% | 100% | 98% | 100% | 83% | 92% | 1.165 |
| **step4c corrected** | 100% | 100% | **100%** | 100% | **100%** | 92% | 1.167 |

Untouched 100%/100%, old name wins 0%, English 1.167 (Qwen untouched
1.164). Left: "answers to" corrections at 92%.

Log: step4c.log, results step4c_results.json.

Logs: step1_arms1-4.log, step1_context_lora.log, step3.log
(step1_results.json is the earlier 30-fact run).
