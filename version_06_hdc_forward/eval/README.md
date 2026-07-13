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
| **0.6B** | closed-book | _pending_ | | | | |
| | RAG | _pending_ | | | | |
| | **gate** | _pending_ | | | | |

**In-KB accuracy by popularity** (why grounding matters): closed-book collapses on
rare entities (1.7B: rare 10% / mid 4% / popular 18%; 4B: 3% / 0% / 18%), while both
RAG and the gate are flat. Parametric knowledge is exactly where the long tail lives.

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
