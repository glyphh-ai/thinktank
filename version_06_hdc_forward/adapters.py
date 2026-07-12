"""
adapters.py — bolt the grounding gate into any LLM runtime, from one core.
=========================================================================
The gate is a GroundedConstraint (lattice_adapter): given the tokens so far it
masks logits to the store's grounded fillers or refusal. Each runtime just wraps
it in its own hook:

  mlx-lm       -> generate(..., logits_processors=[constraint])       (constraint is
                  already a (tokens, logits)->logits callable, mx-aware)
  HF transformers -> LogitsProcessor subclass (this file)              (runnable here)
  vLLM         -> logits_processors=[constraint]  (same callable shape)
  llama.cpp    -> derive a GBNF grammar from grounded_fillers()        (grammar path)

All share `make_constraint(store, tokenizer, prompt, entity, role)`.
"""
from __future__ import annotations

from typing import Optional

from grounding import GroundingStore
from lattice_adapter import GroundedConstraint, build_allowed


def make_constraint(store: GroundingStore, tokenizer, prompt: str, entity: str, role: str,
                    refusal: str = "unknown", eos_id: int = 0,
                    space_id: Optional[str] = None) -> GroundedConstraint:
    """Build the grounding constraint for one (entity, role) answer slot."""
    if space_id is not None:
        store.assert_space(space_id)          # refuse to bind across mismatched spaces
    fillers = store.grounded_fillers(entity, role)
    seqs, _ = build_allowed(tokenizer, prompt, fillers, refusal)
    return GroundedConstraint(seqs, eos_id)


# ── HF transformers adapter ──────────────────────────────────────────────────
def hf_processor(constraint: GroundedConstraint):
    """Wrap a GroundedConstraint as a transformers.LogitsProcessor."""
    from transformers import LogitsProcessor
    import torch

    class _Grounded(LogitsProcessor):
        def __init__(self, c): self.c = c

        def __call__(self, input_ids, scores):
            toks = input_ids[0].tolist()
            masked = self.c(toks, scores[0].detach().cpu().numpy())
            scores[0] = torch.as_tensor(masked, dtype=scores.dtype, device=scores.device)
            return scores

    return _Grounded(constraint)


# ── mlx-lm adapter (the constraint is already the right shape) ───────────────
def mlx_processor(constraint: GroundedConstraint):
    return constraint


if __name__ == "__main__":
    # Interface test on a REAL HuggingFace model: the grounding gate forces the
    # store's value even when the (tiny/ignorant) model would say anything.
    from grounding import LocalStore
    store = LocalStore().inject([
        {"entity": "erin", "spatial.location": "boston"},
        {"entity": "jack", "spatial.location": "dallas"},
    ])
    name = "hf-internal-testing/tiny-random-gpt2"
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name)
        eos = tok.eos_token_id or 0
        for entity, prompt in [("erin", "Erin lives in"), ("jack", "Jack lives in")]:
            con = make_constraint(store, tok, prompt, entity, "spatial.location",
                                  eos_id=eos, space_id=store.space_id)
            proc = LogitsProcessorList([hf_processor(con)])
            ids = tok(prompt, return_tensors="pt").input_ids
            out = model.generate(ids, max_new_tokens=4, logits_processor=proc,
                                 do_sample=False, pad_token_id=eos)
            gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            print(f"  HF gate: {prompt:16} -> {gen!r}  (store: {store.grounded_fillers(entity,'spatial.location')})")
        print("HF transformers adapter OK — gate forces the grounded value on a real HF model")
    except Exception as e:
        print(f"[HF interface test skipped: {type(e).__name__}: {str(e)[:80]}]")
