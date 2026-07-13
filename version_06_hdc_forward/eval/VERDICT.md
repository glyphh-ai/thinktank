# Verdict: does the grounding gate beat a database?

**No — as implemented, the gate IS a database lookup. The "better than a DB" claim
is refuted.** Recorded honestly (2026-07-12) so we don't fool ourselves.

## What the evidence shows

Our `GroundedConstraint` masks the vocabulary to the stored filler tokens (or
abstain), so the output *is* `store[entity][relation]`. Demonstrated: asked a
two-fact question ("who was the composer of X, and what year"), the gate can only
emit the one KB value — it cannot generate the sentence or the second fact.

- Raw LLM: fluent prose, **hallucinates the fact every time** (Rimsky-Korsakov →
  "Hans Zimmer", Hal Kanter → "John Landis").
- Gate: emits exactly the DB value, adds nothing, generates nothing.

So on the axis a DB is built for (structured factual answers) the gate **ties** a DB
and is more complex. On the axis a DB *can't* do (fluent grounded generation) the
gate **also can't do it** — it replaces the answer with the lookup. There is no
capability axis on which the current mechanism beats a database.

The rigorous PopQA result still stands for what it is: a **reliability guarantee for
LLM+KB systems, better than RAG on guarantee + latency**. But that is *not* "better
than a DB", and single-fact QA never was the setting to prove that.

## What would actually beat a DB (and isn't built)

The only thing that beats both a DB (can't generate prose) and a raw LLM
(hallucinates) is **fluent generation with the factual spans grounded** — the model
writes naturally, and every factual assertion is constrained to the KB or forced to
abstain *in place*. Two possible mechanisms, both unbuilt, both real work:

1. **In-loop span gating** — detect each factual-assertion span during generation
   (the v05 role head / an assertion classifier) and constrain only those spans to
   KB-consistent tokens. Hard part = assertion detection in free text (the seam we
   flagged early and never closed).
2. **Generate-then-verify-repair** — free generation, then extract each asserted
   fact, verify against the KB, redact/abstain the ungrounded ones. Softer guarantee
   (bounded by extraction recall), different architecture.

Until one of these exists and is measured on free-form generation, "better than a
DB" is unproven — and the current gate is a DB with extra steps.

## Falsification result: assertion-detection recall (the make-or-break number)

Ran the cheap falsification (`falsify.py`): can we DETECT a factual assertion in
model-written text well enough to gate it? Best case — ONE clean fact per sentence,
strongest local extractor (Qwen3-4B), most generous metric (fact mentioned anywhere):

    surface detection recall = 97.4% (76/78)
    implied P(ALL caught): k=3 -> 92%, k=5 -> 88%, k=8 -> 81%

**97.4% is an optimistic ceiling and it is NOT enough for a hard guarantee.** A
multi-fact paragraph leaks an ungated assertion 12-19% of the time — and this is the
best case. *Gateable* recall (extracting a triple that canonically matches the KB
schema, not just mentioning the value) is lower — even here the extractor flipped
subject/object direction — and on messy free generation it drops further.

Caution learned: our first metric read 66.7% due to an object-column-only match bug;
the assertion WAS detected but direction-flipped. Fixed to 97.4%. (We nearly refuted
on a measurement artifact — the reason to check twice.)

## THE test: gate vs a real database (text-to-SQL), not RAG (`showdown.py`)

Every earlier benchmark compared the gate to RAG (a weak baseline for structured
data). The decisive test adds the real incumbent — an LLM writing SQL over a DB —
across query classes (Qwen3-4B, 60-customer org):

| query class | SQL (DB) | RAG | HDC/gate |
|---|---|---|---|
| structured (agg / count) | exact | — | exact (== DB) |
| fuzzy churn (easy: column proxy) | 100% | 100% | 100% |
| holistic similarity ("like X") | 0%* | 75% | 80% |

\* SQL's 0% was **bad query generation**, not a limit: our HDC similarity IS
categorical-attribute overlap = a `COUNT of matching columns` SQL query; a
well-written one ties HDC.

**There is NO query class where HDC uniquely wins.** Where SQL wins (structured),
HDC ties it (it *is* a DB query). Where SQL stumbles (holistic similarity), RAG
matches HDC. HDC is dominated by "SQL for structured + LLM-over-rows for fuzzy."
Scale doesn't save it: exact queries at scale → the store (a DB) wins (frontier
result); similarity at scale → DB vector search (pgvector); random-atom HDC ≡
categorical overlap ≡ SQL-computable, adding nothing a modern DB lacks.

## Decision (final, evidence-based)

- **Close the lid on "provably better than a DB."** A hard 0-hallucination guarantee
  in free generation requires ~100% assertion detection; the best case is 97.4% and
  reality is worse, so ungated assertions leak. **Refuted, with a number.**
- **The current gate is a database lookup** (replaces the answer with the stored
  value); it ties a DB on structured QA and does not generate grounded prose.
- **What 97.4% DOES enable is SOFT grounding**, not a guarantee: a
  generate-then-verify-repair system could catch ~most hallucinations — a strong
  hallucination *reducer*, competing with RAG, **not** a DB-beater.
- **Honest bottom line:** there is no version of this that beats a database. The
  achievable wins are (a) a provable grounding layer for slot-shaped LLM+KB answers
  (= a DB with NL I/O), and (b) soft hallucination reduction in generation (~RAG).
  If the bar is "beat a DB," the answer is **no** — lid closed.
