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

## Decision

- **Close the lid on "better than a DB" for the current mechanism.** It's a database.
- Keep the honest, narrow result if useful: a provable grounding layer for existing
  LLM+KB (RAG) pipelines — retrofit safety, not a new capability.
- OR commit to building span-level grounded generation (#1 or #2) — a genuine
  research project — and re-test *that* against a DB on a generation task.
