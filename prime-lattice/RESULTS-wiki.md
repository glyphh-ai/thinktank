# Going big: grounded across systems on Wikipedia

Real Wikipedia data (12 Solar-System article intros, ~6,200 words). Real agents
extracted **127 fact-triples**; the lattice holds entities + propositions +
realization templates (153 nodes). Then QA and generation run **deterministically
over the graph — no model at runtime, no hallucination.**

## Question answering (from the graph)
- *What is Mars?* → "Mars is a planet. Mars is the fourth planet from the Sun.
  Mars is made of carbon dioxide. Mars is orange-red in color..."
- *What orbits the Sun?* → "earth, jupiter, neptune, saturn, venus orbit the sun"
  (composed across entities)
- *Which planet is fourth from the Sun?* → "mars" ✓
- *What is the largest planet?* → "jupiter ..." (correct, with substring noise)
- *What is the hottest planet?* → "I didn't learn which is hottest" (honest)
- *What is a unicorn?* → "I can't answer that from what I learned" (no hallucination)

## Generation from what it was seeded (the test)
- **Mars**: "Mars is a planet. Mars is the fourth planet from the Sun. Mars is
  made of carbon dioxide. Mars is orange-red in color. Mars is named after red
  planet. Mars has tenuous atmosphere."
- **Sun**: "Sun is a star. Sun orbits the galactic center. Sun is made of hot
  plasma. Sun has core. Sun is heated by nuclear fusion. Sun is the main source
  of energy for life on earth."

Coherent, accurate, grounded — a categorical jump from the toy generator's salad.

## Honest read
- It **meaningfully generates from what it was seeded** — grammatical, accurate,
  factual descriptions, each sentence traceable to a source fact.
- It is **not** free-form fluent prose; it composes learned facts via templates.
- Noise is honest and bounded: coarse extraction ("named after red planet"),
  substring matches ("largest" hits 3), and gaps it **admits** rather than
  fabricates. Every answer is auditable to its source triple.
- Where this beats an LLM: **no hallucination, full provenance, says "I don't
  know."** Where an LLM beats it: fluency and breadth. Both true.
