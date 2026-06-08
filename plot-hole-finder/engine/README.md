# glyphh engine

Purpose-built hierarchical hyperdimensional (VSA/HRR) memory for plot-hole
detection. Roles → segments → layers → cortex, giving **similarity at every
level** for coarse-to-fine triage of narrative inconsistencies.

This is a standalone implementation specific to this build (pure numpy, no torch
and no dependency on the general glyphh library). See [`../DESIGN.md`](../DESIGN.md)
§5a for the architecture and the rules it follows.

## Layout

```
glyphh/
  hypervectors.py  bind / bundle / permute / similarity primitives (bipolar)
  memory.py        Codebook (symbol -> hv) and CleanupMemory (item memory)
  hierarchy.py     Role, Segment, Layer, Cortex
  resonator.py     resonator network for multi-factor decomposition
examples/
  planted_plothole.py   end-to-end demo on a synthetic 3-chapter story
tests/
  test_glyphh.py
```

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest -q                  # run tests
python -m examples.planted_plothole  # see it localize a planted plot hole
```

## The two rules that make it work

1. **Bundle raw, unbind only at the leaf.** Children are bundled without id
   binding so a parent stays cosine-similar to its parts (similarity at every
   level). Specific items are recovered through each node's cleanup memory, not
   id-unbinding. Value recovery (unbinding) happens only at the segment level;
   layers and the cortex are routing-only.
2. **Store accumulator + vector.** Each node keeps a running integer accumulator
   (pre-threshold, preserves multiplicity — probes read this) and exposes a
   bipolar `vector` (thresholded — cross-node similarity/routing reads this).

## What the demo shows

- **Layer triage:** probing an entity layer for one attribute surfaces every
  value it holds across time; >1 above the noise floor = contested = candidate
  inconsistency — found with no pairwise comparison.
- **Localization:** drilling into the contested layer's segments pins which
  chapters disagree.
- **Coreference:** the cortex routes a stray pronoun mention to the right entity.

The contradiction *decision* (is there a licensing event between the two states?)
is left to the structured store + LLM verifier — the cortex's job is cheap
triage that makes that step rare and local.
