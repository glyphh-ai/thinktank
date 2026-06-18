# version_05_lattice — Ada *in the loop*

Putting [Ada](https://github.com/glyphh-ai/adaL)'s schema-on-write structure
**inside** a model's training loop instead of bolting it on outside — and
measuring, honestly, what that does and doesn't buy.

The baseline is the from-scratch NumPy GPT from `version_0_base` (token+position
embeddings → 1 transformer block → head → cross-entropy → manual backprop →
Adam). Everything here adds to that.

---

## The idea

A normal LM predicts the next token. Here we add a **second head** that predicts
each token's **Ada slot** — one of Ada's fixed 7-layer × 33-role universal
lattice (entity.name, spatial.location, relational.object, …), vendored from
glyphh-ai/adaL (Apache-2.0). The Ada slot is a **target in the loss**:

```
total_loss = LM_cross_entropy  +  λ · ROLE_cross_entropy
```

Why a target and not an input: in `version_04_memory` we fed meaning in as a
*frozen input channel* and the model **ignored it** (tied the baseline). Putting
the structure in the **gradient** instead makes it load-bearing — it can't be
ignored, and the model learns to **emit** the structure itself, in-loop.

---

## Files

| file | what it is |
|---|---|
| `step1_data.py`, `step2_embedding.py`, `step3_gpt.py`, `test_train.py` | the `version_0_base` GPT + benchmark (unchanged) |
| `ada_loop.py` | dual-head model (LM + Ada-role); toy fact corpus; dual-head backprop, gradient-checked |
| `prove.py` | killer-baseline proof: does it learn *structure* or memorize word→role? |
| `big_loop.py` | the same, at scale on Ada's **real** enriched data (3 domains) |
| `trajectory.py` | side-study: the embedding learning-trajectory as a "why" (Markov-chain-of-history) |
| `task3_trust.py` | does in-loop structure buy **calibration / trust**? |
| `long_memory.py` | does it enable **long-horizon recall** a windowed LM can't do? |
| `FINDINGS.md` | full per-experiment detail (this README is the overview) |

Run any with `python <file>.py`. They reuse the gradient-checked block helpers
from `step3_gpt.py`. `big_loop.py` auto-clones glyphh-ai/adaL for its data.

---

## Results

### 1. It works, and it's free (`ada_loop.py`)
3 seeds, toy fact corpus, dual-head backprop gradient-checked (1.2e-7):

```
config                     LM next-word acc   Ada role acc
LM only (λ=0)                    64.0%            4.4%   (role head ≈ chance)
Ada in the loop (λ=1)            63.9%           90.5%   (model LEARNED the slots)
```

Role accuracy 4.4 → 90.5 at **zero LM cost**. The "why" is generated in-loop:
`carol lives in → austin [spatial.location]`, `jack is → fifty [temporal.age]`.

### 2. It learns *structure*, not a lookup table (`prove.py`)
Genuine ambiguity (a city is `spatial.location` after "lives in", `spatial.origin`
after "is from"). Fair tagging; both model and a word→role lookup see the same
token — only the model has context:

```
role accuracy        overall   AMBIGUOUS   unambiguous
lookup baseline        91.8%      58.7%        100.0%
Ada-in-the-loop        99.2%      96.1%        100.0%
```

On ambiguous tokens: **+37.4** over lookup. The only way to win there is to use
context → it learned Ada's context-dependent slot assignment.

### 3. It holds at scale on real data (`big_loop.py`)
14,000 real facts (chemistry / philosophy / law) → 152k tokens, vocab 8k, 5,392
ambiguous words:

```
role accuracy        overall   AMBIGUOUS   unambiguous
lookup baseline        69.8%      63.4%        99.8%
Ada-in-the-loop        90.0%      88.1%        98.8%
```

Ambiguous gap **+24.7** (shrank from the toy's +37 on messier real vocab, but
did not collapse). LM acc 36.3% (λ=1) vs 35.1% (λ=0) → **still free**.

### 4. Does it buy *trust*? — NULL (`task3_trust.py`)
Calibration ablation (does the model get less confident on facts it was never
told?). AUROC: plain LM **0.984** vs Ada-in-loop **0.954** (Δ −0.030). No win —
the baseline is already near-ceiling because softmax confidence *already* encodes
"did I memorize this". Honest negative: the test was too easy for the baseline to
fail.

### 5. Does it do something a windowed LLM CAN'T? — YES (`long_memory.py`)
40 novel facts streamed; the in-loop model extracts each `(entity, slot, value)`
from its role head **as it reads** → persistent store → recall at any distance:

```
recall@          plain LM (context)   in-loop + store
in-window               50%               100%
beyond-window           11%                97%
```

Beyond the context window the plain LM is **blind** (11%, confabulating its stale
trained value); the in-loop model + store recalls **97%** at any distance. Unlike
#4, there's no headroom problem — it's a hard architectural wall.

---

## Verdict (calibrated)

- ✅ In-loop structure is **learnable, free, and scales** (toy → 30× real data).
- ✅ It learns **context-dependent** structure, not memorization (+24–37 vs lookup).
- ✅ It enables a **capability a plain LM can't have** — persistent recall beyond
  the context window — by natively turning generation into structured memory
  (read → emit structure in-loop → persist → recall, end to end).
- ⚠️ It does **not** automatically buy calibration/trust on tasks the baseline
  already handles.

The honest framing: in-loop isn't "a smarter model" — it's **a model that
natively writes its own auditable, persistent memory.** That's Ada's value
(auditable structure + memory beyond the window) fused *into* the model instead
of bolted alongside it.

### Honest decomposition of the recall win
The *recall* itself is external memory — Ada's domain, already proven to 1M
facts; a windowed LM losing to a store is not surprising. The part that's
genuinely new here is the **97% in-loop extraction accuracy**: the model produces
the right triples to populate the store with **no separate enricher pass**.

### Still untested (the road to "proven in the wild")
1. **Messy text.** Extraction here is on clean rendered facts; real text is harder.
2. **Enricher label-noise.** Labels came from Ada's structured fields (clean); the
   real enricher hallucinates a few % — training on noisy targets is untested.
3. **Frontier scale.** This is a ~60k-param NumPy model, not a real transformer.

---

*Ada schema vendored from [glyphh-ai/adaL](https://github.com/glyphh-ai/adaL)
(Apache-2.0).*
