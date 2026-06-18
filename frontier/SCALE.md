# Scaling verdict — exact substrate vs RAG vs bare LLM, 1M → 1B

`scale_bench.py`, pre-registered bars locked before the run. Three information-
access architectures: **Ada** (exact index = int arrays + bincount, what Ada's
count/lookup/version ops reduce to), **RAG** (top-k over a dense embedding index),
**bare** (a context window of W=2000 facts).

```
            recall(ada/rag/bare)  aggregate   refuse    belief    RAG index   Ada store   time
1M            100 / 99 / 0        100/0/0     100/0/0   100/0/0     0.5 GB       2 MB       20s
10M           100 / 98 / 0        100/0/0     100/0/0   100/0/0     5 GB        20 MB      423s
100M          100 / — / 0         100/—/0     100/—/0   100/—/0     102 GB ✗    200 MB     2s
1B            100 / — / 0         100/—/0     100/—/0   100/—/0    1024 GB ✗     2 GB      28s
```
`—` / `✗` = the RAG dense index exceeds memory and cannot be built.

**PRE-REGISTERED VERDICT: PASS at every scale, all four tests.**

## What this proves (cleanly)
- **Recall:** the bare context-window model is blind past its window (0%) at every
  scale; Ada and a *fair* RAG both nail single-fact recall. Recall is the bare wall,
  not RAG's.
- **Aggregate / refuse / belief:** RAG scores 0 (or can't be built) and bare 0, at
  every scale — **architecturally**, not from a weak embedder: you cannot count what
  you can't fully see, cannot refuse a miss if you always return the nearest thing,
  cannot return the current version without versioning. Ada is exact at all scales.
- **The scaling asymmetry is the headline:** past ~10M the RAG index is physically
  impossible (1 TB at 1B), while Ada's store is a 2 GB array answering in seconds.
  At 1B, the exact substrate is the *only thing left standing.*

## What this does NOT prove (scope, not excuses)
- **It's an architecture simulation, not a production run.** Ada-here = numpy int
  arrays (which *is* what Ada's exact ops are); RAG-here = "the index doesn't fit"
  (which *is* real). But Qwen3 + real FAISS + real Postgres weren't run — the GPU
  run adds the NL→op translation layer (equal error across systems).
- **Synthetic facts.** Exactness scales regardless (bincount is exact), but
  real-world accuracy also depends on enrichment quality, which is untested here.
- **Much of this is true by construction.** "An indexed store beats not having one,
  and a 1 TB vector index doesn't fit" is rigorously confirmed and scaled here — but
  it isn't a surprising discovery. The non-obvious parts (does Qwen3 translate NL→ops
  correctly at scale; does the in-loop model populate the store accurately from messy
  text) need the GPU + real-data run.

## Bottom line
The **architectural scaling claim is won to 1B**: the exact substrate answers
recall/aggregation/refusal/belief at a billion facts in seconds and 2 GB, where RAG
is impossible and the bare window never competed. The remaining open work is the
*systemic* claim — the real model driving the real substrate on real data.
