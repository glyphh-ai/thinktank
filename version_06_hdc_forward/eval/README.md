# Guaranteed-grounded generation: a KB-membership decode gate on Qwen3

**Preprint-draft evaluation.** Claim: constraining a stock LLM's decoding to a
knowledge base's stored values gives **guaranteed grounding and provable abstention**
— zero hallucination on out-of-KB queries — matching or beating retrieval on in-KB
accuracy, at a fraction of the latency, independent of model size.

## Setup

- **Data:** [PopQA](https://github.com/AlexTMallen/adaptive-retrieval) (Mallen et al.
  2023) — 14k popularity-stratified Wikidata entity questions. We sample balanced
  across popularity terciles (rare / mid / popular) and split **by entity** into a
  **KB** (facts we store) and a **held-out** set (real facts we do *not* store).
- **Queries:** *in-KB* (fact is grounded) and *out-of-KB* (fact real but absent — a
  grounded system must **abstain**; the model may still know it parametrically).
- **Methods (same stock Qwen3, chat template, greedy):**
  - `closed-book` — parametric only; prompted to answer or say "I don't know".
  - `RAG` — dense retrieval (Qwen3-Embedding-0.6B) over the KB passages, read-then-answer, prompted to abstain if the context lacks the answer.
  - `gate` — probe the KB store for `(subject, relation)`; **constrain** the answer to
    the stored value, or **abstain** structurally when the entity/relation is absent.
- **Metrics:** in-KB accuracy (PopQA alias match), out-of-KB **abstention** and
  **hallucination** (= asserted-and-wrong), **reliability** = 1 − (confidently-wrong
  rate), and latency. N ≈ 150/model (77 KB / 73 held-out).

## Results (Qwen3, 4-bit MLX)

| model | method | in-KB acc | out abstain | out **halluc** | reliability | latency |
|---|---|---|---|---|---|---|
| **1.7B** | closed-book | 10% | 43% | **45%** | 52% | 175 ms |
| | RAG | 79% | 99% | 1.4% | 89% | 249 ms |
| | **gate** | **100%** | **100%** | **0%** | **100%** | **77 ms** |
| **4B** | closed-book | 7% | 79% | 16% | 85% | 277 ms |
| | RAG | 100% | 100% | 0% | 100% | 388 ms |
| | **gate** | **100%** | **100%** | **0%** | **100%** | **115 ms** |
| **0.6B** | closed-book | 2.6% | 90% | 8% | 89%* | 135 ms |
| | RAG | 84% | 100% | 0% | 99% | 195 ms |
| | **gate** | **100%** | **100%** | **0%** | **100%** | **63 ms** |

\* **Reliability is gameable by abstaining.** Closed-book at 0.6B scores 89%
"reliability" only because it abstains on ~everything (2.6% in-KB accuracy — it says
"I don't know" to nearly all questions). Reliability must be read *jointly* with
coverage/accuracy. The **gate is the only method that hits 100% reliability AND 100%
in-KB accuracy AND the lowest latency at every model size.**

**In-KB accuracy by popularity** (why grounding matters): closed-book collapses on
rare entities (0.6B: 3/4/0%; 1.7B: 10/4/18%; 4B: 3/0/18%), while both RAG and the gate
are flat across the tail. Parametric knowledge is exactly where the long tail lives.

**Size-independence:** the gate is flat **100% / 100% / 0%** across 0.6B→4B; closed-book
and (small-model) RAG are not. The guarantee does not depend on model capability.

## Findings (honest)

1. **Closed-book hallucinates heavily** on out-of-KB (45% at 1.7B) and is near-useless
   on rare entities — the motivating failure.
2. **RAG is a strong baseline**, and at 4B it *matches* the gate (perfect reading,
   near-perfect abstention). But two gaps persist at every size:
   - **Guarantee.** RAG's 0% hallucination is *empirical*, prompt-dependent, and leaks
     (1.4% at 1.7B). The gate's is *architectural* — it cannot emit an ungrounded token.
   - **Latency.** The gate is **3× faster** (no retrieval; constrained decode locks onto
     the short stored value + stop, instead of reading a long context).
3. **Even with the right passage in context, a small LLM misreads it** (RAG 79% in-KB at
   1.7B). The gate removes that failure mode by construction.
4. **The guarantee is size-independent** (gate: 100%/0% at all sizes) while closed-book
   and small-model RAG are not.

## The hardest objection — "why not just query the database?"

This is the critique that would sink the paper if unaddressed, so we address it directly.

On single-fact KBQA the gate emits `store[entity][relation]`, so a reviewer can say:
*"you've shown a constrained decoder equals a database on database questions — the LLM
is a costume; just query the DB."* Our honest answer, three parts:

1. **PopQA isolates the guarantee; it is not the target use case.** We test the safety
   property on a clean factual benchmark the way a seatbelt is tested on a crash dummy —
   not because dummies drive, but because it isolates the property. The gate's *value*
   is in **grounded generation** (an LLM composing free-form text that must not assert
   ungrounded facts), which single-fact QA under-tests. That generative eval is the
   needed next experiment (below) and we do **not** claim the QA result stands in for it.
2. **The fair comparison is gate vs RAG, not gate vs DB.** "Why use an LLM over a KB at
   all?" applies *equally* to RAG — and the field already answered it (NL interface,
   generative flexibility, unstructured inputs). Given a system already committed to an
   LLM+KB (i.e., every RAG deployment), the gate is a strictly better grounding
   mechanism: a **hard** guarantee where RAG's is empirical, **no misread failures**, and
   **lower latency**. It is a drop-in reliability upgrade for LLM-over-KB systems.
3. **We do not claim the model "knows" more.** The gate converts KB correctness into
   output *with a guarantee*. It is a reliability result, not a knowledge result.

**What we must NOT claim** (and don't): that the gate beats a database at KBQA, or that
this QA benchmark demonstrates value a database couldn't provide. It demonstrates the
*guarantee* and quantifies it against the standard LLM+KB baseline (RAG).

## Refutations we ran (adversarial self-checks) — Qwen3-1.7B, N=360, 95% CIs

Rigorous run (181 KB / 179 held-out), Wilson intervals:

| method | in-KB acc [95% CI] | out abstain | out halluc | reliability |
|---|---|---|---|---|
| closed-book | 7.7% [5–13] | 38% | 52% | 46% |
| RAG | 73.5% [67–79] | 99% | 0.6% | 86% |
| **gate** | **100% [98–100]** | 100% | 0% | 100% |
| **gate_linked** (no gold) | **100% [98–100]** | 100% | 0% | 100% |

- **"Gold-slot cheat" — REFUTED.** `gate_linked` extracts `(entity, relation)` with a
  regex (no gold): slot recovery **100%** on 600 items, and its accuracy **equals `gate`
  (100% = 100%)**. Honest caveat: this holds *because PopQA embeds the entity in the
  question*; free-form entity linking is a separate, genuinely hard problem.
- **"In-KB=100% is trivial" — REFUTED.** RAG **retrieved the gold passage on 100%
  [98–100]** of in-KB queries yet answered only **73.5%** correctly — a **26.5%
  pure-misreading gap** (retrieval was perfect; the model ignored/garbled the answer it
  was handed). The gate closes that gap by construction. The in-KB result guards a real,
  measured failure mode — not a lookup tautology.

## Limitations (state these in the paper)

- **Oracle slots.** The gate probes with gold `(subject, relation)`; it assumes entity
  linking + relation classification. RAG uses only the question. A fair follow-up runs
  the gate with a learned linker/router (we have a semantic relation router,
  `schema_guard.py`) and reports the degradation.
- **The gate's in-KB accuracy is the KB's correctness by construction** (KB = gold). The
  claim is a *guarantee that converts KB correctness into output*, not that the model
  "knows" more. Framed as reliability, not capability.
- **Coverage cost.** On out-of-KB facts the model *does* know, the gate abstains — trading
  coverage for the reliability guarantee. Correct behavior for grounded/enterprise QA;
  a cost for open-domain.
- **N and CIs.** Report per-model N and Wilson intervals; bump N for the camera-ready.

Reproduce: `python run_eval.py --model mlx-community/Qwen3-1.7B-4bit --per-tercile 50`
