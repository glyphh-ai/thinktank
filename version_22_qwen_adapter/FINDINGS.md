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
| **7b: points bundled per rule token** | **18%** | **28%** | 0% | 40% |

7b (decided after 7a failed, so the second try on the test questions):
every point that chose the same token is summed into one glyphh, so what
the situations share (a sentence just closed, about to end) adds up and the
reply words wash out: 59 " Cheers" points become one key. Sign-off goes
0% -> 18% (7 of 40) with nothing in the prompt and unrelated answers intact ("Boil
an egg for 10-12 minutes ... Cheers!"). The ban learns nothing: Qwen's
practice questions were about the rule ("What are the advantages of
decision trees over logistic regression?"), never a request for a model, so
the rule never acted where the test asks. Next: practice questions that
put the model where the rule acts, not questions about the rule.

Logs: step7a.log / step7b.log, results step7a_results.json / step7b_results.json.

## step8 — predict the reply, then look back at what the rule changed

step8_lookback.py: Qwen drafts the whole reply with no rules (the
prediction). Each spot of the draft is a situation (the next 3 draft
tokens, END included, + the last 3 reply tokens); the nearest stored moment
above tau says "about to say X here, the rule said Y": X is pushed down and
Y up by their measured log-ratios, the reply is cut there and drafted
again. Rehearsal is the same loop with the rules as teacher, so the stored
moments are Qwen's own unruled drafts with earlier corrections applied;
moments with the same (X, Y) are bundled. Settings fixed before 8a.

| arm | ends with "Cheers!" | mentions "Cheers!" | ban | says logistic |
|---|---|---|---|---|
| Qwen alone | 0% | 0% | 0% | 40% |
| rules in the prompt | 100% | 100% | 95% | 5% |
| 7b (live nudges) | 18% | 28% | 0% | 40% |
| **8a: look back on the draft** | **30%** | **100%** | 0% | 40% |
| 8b: tempting practice + bundles by Y | 2% | — | 0% | 40% |
| 8c: tempting practice, (X, Y) bundles | 25% | 100% | 0% | 40% |

- **The sign-off now always fires**: "about to end -> Cheers" (51 moments,
  one key) matches the END of every draft, 40 of 40, and the answer before
  it is word for word Qwen's own in 15-17 of 20 general questions. It only
  *ends* there 25-30% of the time: with no rule in view Qwen carries on
  after it (an emoji in 12-13 replies, "**Answer:** ..." in 16-17).
- **The ban still learns nothing, and now for a different reason.** 8a's
  practice had no requests for a model again. 8c's did ("They want to
  predict if a customer will default on a loan."; ask: "requests that would
  tempt an assistant to do what the instruction forbids"), and rehearsal
  stored ' logistic' -> ' decision' x8 and ' logistic' -> ' a' x6. At test
  they fire (5 of the 8 drafts that say logistic are changed), but the
  redraft has no rule in view and walks back: "Use a logistic regression
  model or a decision tree" -> "Use a classification model like logistic
  regression or a decision tree". All 8 still say logistic.
- 8b (bundle by Y, push down whatever the draft was about to say) wrecks
  replies: a loose match pushes the draft's token down and something
  other than Y wins.

So predicting the run finds *where* a rule applies (100% for the sign-off,
5 of 8 for the ban), but a one-token correction doesn't change what the
model means to say next. 8b and 8c were decided after 8a on the same test
questions: exploration, not a result. No beam was run: there was nothing
to pick between, every corrected draft fails the same way. Next: keep the
look-back as the trigger and let what it recalls stay in view for the
redraft (the rule that made the stored moment), instead of a token push.

### 8d / 8e — halt, restart, halt again, every time the prediction matches a gate

In 8a-c a gate could halt a reply once. 8d (--rehalt) lets it halt every
time the redraft matches it again (it only sits out while its own token is
among the last 3 written). 8e adds a tau per gate (--per-gate).

| arm (8c's practice and bundles) | ends with "Cheers!" | mentions | ban | says logistic | halts per reply |
|---|---|---|---|---|---|
| 8c: each gate once | 25% | 100% | 0% | 40% | 2.5 |
| 8d: gates re-halt | 25% | 100% | 0% | 40% | 3.2 |
| 8e: re-halt + a tau per gate | 8% | 45% | 15% | 25% | 7.9 |

- 8d changes nothing for the ban. diag_rehalt.py: at every ' logistic'
  left in a reply the ' logistic' gate IS the nearest gate, at 0.52-0.68,
  just under the one tau (0.684), which is set by gates on common words
  (',' -> '.', ' to' -> '!').
- 8e gives each gate its own tau (0.19-0.66). The ' logistic' gates now
  halt again and again, and the ban moves for the first time (0% -> 15%,
  says logistic 40% -> 25%), but the swap is a token, not a meaning: Qwen
  answers with " Logistic" (capital, another token), and the junk gates
  rehearsal also stored ('They' -> 'I', '<end>' -> ' \n\n') fire at their
  lower taus and wreck the replies ("ICheers! To sleep better..."): 2 of
  20 general answers keep Qwen's own wording, sign-off 8%.

So the halting sequence works as a loop; what fails is what is done at each
halt. A token swap can't hold a rule ("logistic" -> "Logistic"), and gates
bundled from single-token differences include noise that a per-gate tau
lets through. Five tries on these test questions now: exploration. Next
is unchanged: at a halt, bring back the rule that made the stored moment
and redraft with it in view; and keep only gates that trace to a rule.

Logs: step8.log / step8b-e.log, results step8_results.json /
step8b-e_results.json; diag_asks.py (the practice ask), diag_rehalt.py.

## step9 — at a halt, the rule itself comes back into view

step9_recall.py: step8's predict / look back / halt loop, but a halt no
longer swaps a token. Each stored moment knows which rule made it (in
rehearsal, the rule that alone explains > 8 nats of the move; a spot no
single rule explains is not stored); at a halt that rule's sentence goes
into view and the reply is drafted again from there. A rule in view has no
gates left, so at most one halt per rule. A tau per gate. Settings fixed
before the run; 40 NEW test questions (never used), the old 40 beside them.

| new questions | ends with "Cheers!" | ban | says logistic |
|---|---|---|---|
| Qwen alone | 0% | 5% | 25% |
| rules in the prompt | 100% | 90% | 5% |
| **recall at a halt (nothing in the prompt at the start)** | **100%** | **75%** | 20% |

| old questions (used since step5) | ends with "Cheers!" | ban | says logistic |
|---|---|---|---|
| Qwen alone | 0% | 0% | 40% |
| rules in the prompt | 100% | 95% | 5% |
| recall at a halt | 100% | 55% | 45% |

- First time the rules carry with nothing in the starting prompt: sign-off
  100%, ban 75% on fresh questions. Several "says logistic" replies are
  the rule spoken aloud ("Use a decision tree ... Logistic regression is
  not suitable"), which the ban check counts as a miss.
- **But the gates are not selective, so this is not yet the memory knowing
  when.** All three rules were recalled on every one of the 80 replies,
  the ban on "Why do we yawn?" included. With a tau per gate (0.12-0.43)
  and gates such as 'They', ' use', ' can', ' historical', something
  always matches. What ran is close to "rules in the prompt", reached by a
  halt, and it does a little worse than the prompt (rules arrive
  mid-reply).
- There is no ' logistic' gate at all: with per-rule attribution the ban's
  63 moments scatter over many tokens and only 5 small bundles survive.

So the action at a halt is right (recall the rule, don't swap a token);
the open part is the gate. Next: gates that fire only where their rule
acts (the ban on ML questions, not on general ones), measured as "rule in
view, general vs ML", on a dev set of questions, not on these 80.

Log: step9.log, results step9_results.json.

## step10 — gates that fire only where their rule acts

step10_gates.py: step9's rehearsal and recall, two changes to the gate.
The key gains a gist (a glyphh of the content words of the question and
the predicted reply), since the prediction is the whole draft, not three
tokens of it. And a gate's tau comes from rehearsal itself: the 1,317 spots
of practice drafts where its rule was not in view and the rules changed
nothing (99.5%), not from Frankenstein. Four gate variants were compared on
a dev set (the old 40 questions) and one picked before any test run:
gist + one gate per (rule, token about to be said), 10 gates.

| 40 questions never used | ends with "Cheers!" | ban | says logistic | ban rule in view, general / ML |
|---|---|---|---|---|
| Qwen alone | 0% | 0% | 25% | — |
| rules in the prompt | 100% | 95% | 0% | 100% / 100% |
| step9 gates (other questions) | 100% | 75% | 20% | 100% / 100% |
| **step10 gates** | **100%** | 15% | 15% | **0% / 15%** |

(step9's 40 questions, used once: 100%, ban 20%, in view 0% / 15%.)

- **The sign-off is done, and selectively.** Nothing in the prompt; Qwen
  writes its own answer; the gate halts at the predicted end, the sign-off
  rule comes into view, the reply ends "Cheers!". 100% on unseen questions,
  one halt, the answer before it untouched.
- **The gates are selective now**: the ban is never recalled on a general
  question (step9: always). "Call me Chris" 0% general too.
- **But the ban gate fires on only 15% of the ML questions**, so the ban is
  15%. Where it fires it works ("Use a decision tree to predict whether a
  student drops out. Cheers!"). The ban's 63 rehearsal moments scatter over
  what the draft was about to say (' use' 7, ' can' 5, ' historical' 4,
  'We' 3; no ' logistic' gate), so its gates are small and their taus,
  set against real chat negatives, are high (0.50-0.77).
- Dev told the same story: one gate per rule gave ban 0%, gist or not.

So: predict, halt where a gate matches, recall the rule, redraft. For a
rule that acts at one kind of spot (the end of a reply) that is solved.
For a rule that acts on a topic (any request for a model) the gate does
not yet cover the topic. Next: more rehearsal where the ban acts (the 22
tempting requests are all the ban has), and a ban gate keyed on the gist
alone, so that it covers the request and not a token.

Log: step10.log, results step10_results.json.

## step11 — a gate on the whole predicted reply, for rules that act on a topic

step11_draft_gate.py: step10 unchanged, plus one DRAFT gate per rule, asked
once per predicted reply: does its gist (content words of the question and
the draft) look like the practice drafts the rule went on to act on, or
like those it left alone? If so the reply halts before its first word, the
rule comes into view, and the reply is drafted again. A rule without 10
drafts of each kind gets none: the sign-off acts on every reply, has no
negatives, and stays step10's spot gate at the predicted end. Four variants
on dev (the 80 used questions); picked before the test run: key =
positives minus negatives, tau = the negatives' 99.5%.

| 40 questions never used | ends with "Cheers!" | ban | says logistic | ban rule in view, general / ML |
|---|---|---|---|---|
| Qwen alone | 0% | 0% | 25% | — |
| rules in the prompt | 100% | 85% | 10% | 100% / 100% |
| step10 gates (other questions) | 100% | 15% | 15% | 0% / 15% |
| **step11: + draft gate** | **100%** | **50%** | 25% | **0% / 60%** |

- The ban goes 15% -> 50% on unseen questions and stays selective: never
  recalled on a general question, recalled on 12 of 20 requests for a
  model. Where it is recalled the reply follows it: 10 pass the check and
  2 say it aloud ("Logistic regression is not suitable for this task; a
  decision tree is recommended"), which the check counts as a miss. The 8
  it misses are where the draft doesn't read like a modelling request
  ("Check the bus's schedule...", "Look at the driver's past tips...").
- Sign-off 100% again, one halt at the predicted end.
- The plain bundle of positives does nothing (dev: same as no draft gate):
  the positives are dirty. Rehearsal says the ban "acted" on 63 of 119
  practice questions, "What should I do if my phone is frozen?" included,
  because with the rule in view Qwen sometimes changes an unrelated reply
  (or recites the rule). Subtracting the negatives is what finds the topic.
- Dev, midpoint tau: ban 52% but the ban in view on 15% of general
  questions; the pick trades that recall for 0%.

Two kinds of gate now, both learned in rehearsal: a spot gate for a rule
that acts at a place, a draft gate for a rule that acts on a topic. Next:
clean positives (count the ban as acting only where the reply with the
rule and the reply without it differ in what they recommend), and more
practice requests, to close 50% -> the prompt's 85%.

Log: step11.log, results step11_results.json.

### step11 on Qwen3-14B — the rules, on a model eight times the size

Same script, `QWEN=Qwen3-14B-4bit` (4-bit, MLX). Everything is learned
again from scratch by the bigger model: it writes its own 213 practice
questions, rehearses (23 minutes on an M2 Max; 2 at 1.7B), the draft gate
is picked on the same dev questions (midpoint this time). The test set is
step11's (its second use, first for this model).

| 40 questions | ends with "Cheers!" | ban | says logistic | ban rule in view, general / ML |
|---|---|---|---|---|
| 14B alone | 0% | 0% | 40% | — |
| 14B, rules in the prompt | 100% | 100% | 0% | 100% / 100% |
| **14B, recall at a halt** | **100%** | **65%** | **0%** | 15% / 65% |
| 1.7B, recall at a halt | 100% | 50% | 25% | 0% / 60% |

- It carries to the bigger model and gets better: ban 50% -> 65%, and
  every reply that had the ban in view followed it (13 of 13; at 1.7B two
  recited the rule instead). No reply says logistic.
- The whole gap to the prompt is the gate, not the model: the 7 misses are
  the 7 requests where the ban was not recalled.
- Selectivity slipped: the ban was recalled on 3 of 20 general questions
  (the dev pick was the looser midpoint tau). It did no harm there.
- With 13 quiet practice drafts the sign-off now also gets a draft gate
  and comes into view before the first word; 100% either way.

Log: step11_14b.log, results step11_14b_results.json.

### step4c on Qwen3-14B — the fact memory, on the bigger model

step4c's run unchanged (3,000 facts taught once, two animals per person,
five untaught rewordings, a third corrected once), `QWEN=Qwen3-14B-4bit`,
scored on 80 sampled facts (300 at 1.7B) to keep the run under an hour.

| glyphh memory, 3,000 facts | taught | qa | called | belongs | ask | answers | English |
|---|---|---|---|---|---|---|---|
| 1.7B recall | 100% | 100% | 100% | 100% | 100% | 99% | 1.167 |
| **14B recall** | 100% | 100% | 100% | 100% | 100% | 95% | 0.938 |
| 1.7B corrected | 100% | 100% | 100% | 100% | 100% | 92% | 1.167 |
| **14B corrected** | 100% | 100% | 100% | 100% | 100% | 90% | 0.939 |

Untouched facts 100% / 100%, the old name wins 0%, as at 1.7B. The same
memory, the same tau (0.449), nothing retuned for the bigger model. The
weak wording is the same one ("answers to"). English is bits/char with
the memory on; the 14B's own figure without it was not run here
(--no-before), so the drift is not measured, only that teaching and
correcting moved it 0.938 -> 0.939.

Log: step4c_14b.log, results step4c_14b_results.json.

Logs: step1_arms1-4.log, step1_context_lora.log, step3.log
(step1_results.json is the earlier 30-fact run).
