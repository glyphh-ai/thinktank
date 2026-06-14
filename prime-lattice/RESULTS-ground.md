# Grounded understanding — first measured result

The experiment from the design: take a corpus, break every sentence into the
lattice as **facts**, then measure understanding by what can be **composed**
from them — questions that require chaining 2+ facts (so we measure
understanding, not memory). **Real agents do the extraction.**

## Setup

- **Corpus**: a controlled world (people move between places, carry objects),
  generated with hidden ground truth. Each question — *"Where is the X?"* —
  needs **two supporting facts**: who took X, and where they then went
  (bAbI-task-2 style). Agents never see the answer.
- **Pipeline**: sentence —(real LLM agents)→ structured fact → lattice
  (entities=primes, facts=programs, story=system); question —(deterministic
  composition)→ replay facts to world-state → answer. **No model at answer time.**

## Result

| decomposer | stories | facts in lattice | understanding (composition acc.) |
|---|---|---|---|
| reference parser (ceiling) | 172 | 1108 | **100%** |
| **3 real agents (parallel)** | 30 | 198 | **100% (30/30)** |

The real agents (`prime-lattice/data/agent_facts_30.json`) matched the
reference ceiling: every fact extracted correctly, every two-fact question
answered by composition.

## What this does and does not show

**Shows** (honestly): the loop works end to end — real LLM agents do the
open-ended NL→fact work at build time; facts ground into one schema-on-write
lattice; deterministic composition reconstructs world-state and answers
multi-fact questions with **zero model at runtime**; and it parallelises across
agents (the swarm). Understanding is *measured*, not asserted.

**Does not show yet**: scale (30 stories, not 10k–100k — same pipeline, more
agents); harder grounding (this world is closed-vocabulary; open-domain text
will surface extraction errors and ambiguity the agents must resolve); and
deeper composition (2-fact chains; real understanding needs longer, cross-story
inference). Those are the next stakes.

## Reproduce

```
python prime-lattice/ground.py                                  # reference ceiling
python prime-lattice/ground.py --agent-facts prime-lattice/data/agent_facts_30.json
```
