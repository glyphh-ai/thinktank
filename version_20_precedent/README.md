# version_20_precedent — a model that argues with its precedents

The idea (Chris, 2026-11-06): take Ada's precedent engine and build a kind of
LLM nobody has. Weights hold the skill of reasoning from precedent, Ada holds
every decision.

> **Does training a small LLM to name its legal move (apply, distinguish or
> overrule a precedent) beat training it on the decision alone, on domains
> and policies it has never seen?**

## Prior art (checked 2026-11-06, four searches, not exhaustive)

| work | what it does | what it leaves open |
|---|---|---|
| Memento (2508.16153) | Case bank with outcomes beside a frozen LLM, learned retrieval | The LLM never reasons about whether a case still binds |
| Limited Memory LMs (2505.15962) | Facts out of the weights, forget by deleting | Facts, not decisions; no precedent logic |
| AgentMemoryBench (ICLR 2026) | Measures conflict resolution over time | A benchmark, not a training objective |
| Overruling evals (2510.20941), LegalBench | Test whether LLMs spot overruling | Evaluation only |
| LawLLM (2407.21065) | Trained to recommend precedents | Picks a case, does not apply, distinguish or overrule it |

Open, as far as we can find: the move itself as the training target.

## The test

Each example is a fresh world: invented attributes, invented decisions, a
hidden policy that may change partway through, and clerks who sometimes get it
wrong (the outcome says so). The model sees retrieved precedents and a new case.
An ideal observer enumerates every rule consistent with the precedents, and an
example is kept only if the answer is fully determined by what is on the page.
So 100% is reachable, and no vocabulary is shared between train and test.

Arms, same examples, same steps:

- **decision**: prompt → decision
- **move**: prompt → move, then decision
- **after**: prompt → decision, then move (same labels, no thinking room;
  separates "the labels teach" from "the extra tokens think")

Baselines: nearest precedent, nearest with outcome flip, latest precedent,
zero-shot Qwen3, a frontier model on a sample.

## Predictions, written before any run

1. Nearest precedent does badly on the two hard moves, since those are built
   to fool it.
2. `move` beats `decision` by at least 10 points on unseen domains. If it
   doesn't, the objective is not worth anything and the idea dies here.
3. `move` beats `after`. If they tie, the labels teach and the chain does not.
4. A frontier model reading the same page is hard to beat. If it wins by a
   lot, the only remaining claim is size and cost.
