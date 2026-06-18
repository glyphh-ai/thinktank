# frontier — Qwen3 × Ada-in-the-loop (Colab/GPU)

The frontier-scale version of `version_05_lattice`: take a real **Qwen3** and add
Ada's schema **in the loop** (dual objective = next-token LM + Ada-role token
classification), to test whether the in-loop result holds beyond the NumPy toy.

> **Not runnable on CPU.** The pipeline *logic* is validated by `smoke_test.py`
> (passes here). The Qwen3 run needs a Colab GPU.

## Open it in Colab from GitHub
**Easiest (handles the slashed branch reliably):** Colab → *File → Open notebook
→ GitHub* → paste `timmetim/thinktank` → pick branch `claude/ada3-status-ppr17e`
from the dropdown → open `frontier/ada_qwen.ipynb`. Then *Runtime → Change
runtime type → GPU*.

**Direct link:**
`https://colab.research.google.com/github/timmetim/thinktank/blob/claude/ada3-status-ppr17e/frontier/ada_qwen.ipynb`

If `timmetim/thinktank` is **private**, tick "Include private repositories" in the
Colab GitHub tab and authorize, or just upload `ada_qwen.ipynb` manually.

## What you need
- **Colab Pro+** (A100 40GB → Qwen3-8B) or **Pro** (L4 → ~4B); free T4 → 1.7B.
- **HuggingFace token** (optional) as Colab secret `HF_TOKEN` — avoids download throttling.
- **No Anthropic key** — we use adaL's existing enriched data (39k facts).

## What runs (notebook order)
1. clone repo, `%cd frontier`
2. `pip install transformers peft accelerate bitsandbytes datasets`; GPU check
3. `prepare_data.py` — render Ada's real facts → text + per-token Ada-role labels + QA
4. `smoke_test.py` — CPU sanity check of the dual-objective loop (pre-flight)
5. `finetune_qwen.py --lam 0.5` (Ada in the loop) and `--lam 0.0` (plain LM)

## Files
| file | role | validated here? |
|---|---|---|
| `prepare_data.py` | adaL facts → training/QA data | ✅ ran on real data (31k facts) |
| `smoke_test.py` | tiny-model dual-objective loop | ✅ passes (role acc 87%) |
| `finetune_qwen.py` | Qwen3 + LoRA + role head + dual loss | ⚠️ Colab GPU only |
| `ada_qwen.ipynb` | Colab orchestrator | ⚠️ Colab GPU only |

## The two arms
- **Arm B (this notebook):** does the in-loop role objective hold on a real Qwen3
  at scale, and stay free on LM loss? Compare `--lam 0.5` vs `--lam 0.0`.
- **Arm A (extend adaL `benchmark/`):** Qwen3 + Ada *store* vs RAG on QA over
  1K→1M(→100M) facts — the memory/recall claim with Qwen3 as the answerer. Reuse
  the pre-registered harness in `benchmark/phase3/`, swapping Qwen3 in.

*Ada schema + data from [glyphh-ai/adaL](https://github.com/glyphh-ai/adaL) (Apache-2.0).*
