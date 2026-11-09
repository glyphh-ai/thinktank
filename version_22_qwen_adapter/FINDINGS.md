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

## step5 — instructions across sessions: the fact memory does NOT carry them

step5_instructions.py: session 1 gives three rules once ("Call me Chris.",
"End every answer with 'Cheers!'", "Never recommend logistic regression;
recommend a decision tree instead.") and Qwen answers six questions with
the rules in view; that transcript is learned into the step4c glyphh memory.
Session 2: 40 new questions (20 general, 20 ML), no rules in the prompt.

| arm | name | sign-off | ban (says decision tree, no logistic) | says logistic |
|---|---|---|---|---|
| Qwen alone | 0% | 0% | 0% | 40% |
| rules in the prompt | 0% | 100% | 95% | 5% |
| glyphh memory (session 1 learned) | 0% | 0% | 30% | 30% |

The memory fires on 81% of reply tokens (tau 0.267 from book text against a
208-key chat store) and replays session-1 sentences into unrelated answers
("How long should I boil an egg?" -> "Try a decision tree to analyze text
data and identify spam patternsCheers..."). Nearest-match on the last 8
words recalls what came next, which is right for a fact and wrong for a
rule: a rule has to act on replies whose words share nothing with where it
was said. Instructions need their own channel. ("Call me Chris" is never
followed even in the prompt, so that rule tests nothing yet.) English
1.129 -> 1.142 bpc on this run.

Log: step5.log, results step5_results.json.

## step6 — a rule channel: nudges learned from where the rules decided

step6_rules.py: rerun each session-1 reply through Qwen without the rules;
where the rules made the chosen token > 8 nats likelier is a decision point
(17 found: "...\n\n" -> "Cheers" +19..26, "Use a" -> " decision" +9..17,
sentence end -> "\n\n" +9..16, plus a few knock-on style picks). Key = the
situation (glyphh of Qwen's own top-5 next tokens + the last 3 tokens of
the reply); value = the token, applied as its measured log-ratio; tau from
Frankenstein (0.644); each point at most once per reply.

| arm | sign-off | ban | says logistic |
|---|---|---|---|
| Qwen alone | 0% | 0% | 40% |
| rules in the prompt | 100% | 95% | 5% |
| rule nudges (no rules in prompt) | 0% | 10% | 35% |

Clean now (unrelated answers untouched, unlike step5), but it barely acts:
six demonstrations give 17 situations, and new replies rarely land on one
(the stored "about to recommend" spots had Qwen about to say "churn",
"machine", "simple", never "logistic"). Three variants were tried on the
test questions, so treat these as exploration, not a result.

Log: step6.log, results step6_results.json.

## step7 — rehearsal: the rules practised on Qwen's own questions

step7_rehearse.py: when the rules are given, Qwen writes its own practice
questions (40 everyday + 30 per rule, "where this instruction would
matter"; 72 kept, none equal to a test question), answers them with the
rules in view (signs off 94%), and step6's decision points are harvested:
188 of them. step6 settings, fixed before the run.

| arm | ends with "Cheers!" | mentions "Cheers" | ban | says logistic |
|---|---|---|---|---|
| Qwen alone | 0% | 0% | 0% | 40% |
| rules in the prompt | 100% | 100% | 95% | 5% |
| 7a: 188 points, each its own key | 0% | 23% | 0% | 30% |
| **7b: points bundled per rule token** | **23%** | **28%** | 0% | 40% |

7b (decided after 7a failed, so the second try on the test questions):
every point that chose the same token is summed into one glyphh, so what
the situations share (a sentence just closed, about to end) adds up and the
reply words wash out: 59 " Cheers" points become one key. Sign-off goes
0% -> 23% with nothing in the prompt and unrelated answers intact ("Boil
an egg for 10-12 minutes ... Cheers!"). The ban learns nothing: Qwen's
practice questions were about the rule ("What are the advantages of
decision trees over logistic regression?"), never a request for a model, so
the rule never acted where the test asks. Next: practice questions that
put the model where the rule acts, not questions about the rule.

Logs: step7a.log / step7b.log, results step7a_results.json / step7b_results.json.

Logs: step1_arms1-4.log, step1_context_lora.log, step3.log
(step1_results.json is the earlier 30-fact run).
