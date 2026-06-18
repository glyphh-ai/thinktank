# messy — the real-world messy-text test

The honest open item from `version_05_lattice/FINDINGS.md`: every in-loop result
so far used **clean** text where the slot value appears verbatim (so it's token-
tagging). Real prose is **messy** — the value is paraphrased or implicit
(*"the Texas capital"* for `location=Austin`), so there's no span to tag. The
task becomes **extraction/generation**: messy prose → Ada slots, i.e. *being the
enricher.* This harness tests exactly that.

## Pipeline
```
fetch.py    real messy Wikipedia prose            -> messy.jsonl {title,text}   [no key]
enrich.py   Ada's REAL enricher labels the gold   -> gold.jsonl {title,text,slots}  [NEEDS ANTHROPIC_API_KEY]
eval_messy  extract slots from prose, score vs gold (slot-F1)                     [GPU]
            - zero-shot Qwen3 (prompted)   vs   - in-loop fine-tuned Qwen3
```

## What's validated here vs. what needs key/GPU
| step | status |
|---|---|
| `fetch.py` real-prose acquisition | ✅ ran (12/12 real summaries) |
| `eval_messy.py` slot-F1 metric | ✅ self-test passes (1.0 / 0.75 / 0.0) |
| `enrich.py` (Ada enricher gold) | ⚠️ needs `ANTHROPIC_API_KEY` |
| Qwen3 generations | ⚠️ needs a GPU (Colab) |

## The question it settles
1. Does the clean-data win (role/slot acc 88–99%) **survive on messy real prose**?
2. Does **in-loop fine-tuning** make the model a better enricher than **prompting alone** (zero-shot Qwen3)?
3. How close does either get to **Ada's own enricher gold** (the ceiling)?

This is the test where the value could genuinely drop — paraphrase/implicit slots
are hard. Pre-register the bar before running (e.g. "in-loop slot-F1 ≥ zero-shot
+0.10, and ≥ 0.6 absolute") so it's a clean win/loss.

## Run
```bash
python fetch.py --titles /tmp/adaL/benchmark/curriculum/agi/chemistry/phase1_output/corpus_titles.txt --n 1000
ANTHROPIC_API_KEY=sk-... python enrich.py            # gold (costs ~$0.001/text)
python eval_messy.py --selftest                       # metric sanity (CPU)
python eval_messy.py --gold gold.jsonl --model Qwen/Qwen3-8B   # zero-shot baseline (GPU)
```

*Gold = glyphh-ai/adaL's UniversalEnricher output (Apache-2.0). messy.jsonl /
gold.jsonl are gitignored (regenerate with fetch/enrich).*
