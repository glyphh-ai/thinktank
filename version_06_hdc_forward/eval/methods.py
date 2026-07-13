"""
methods.py — the three answering methods under test, on a stock Qwen3.
=====================================================================
  closed_book — parametric only; may answer or abstain (prompted to).
  rag         — dense-retrieve KB passages (Qwen3-Embedding), read, answer/abstain.
  gate        — our grounding gate: probe the KB store for (subj, prop); constrain
                the answer to the stored value, or ABSTAIN (structural, guaranteed).

All three receive the same question and may abstain; only the gate's abstention is
architectural (it cannot assert an ungrounded value).
"""
from __future__ import annotations

import sys
import os
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grounding import LocalStore                         # noqa: E402
from lattice_adapter import GroundedConstraint, build_allowed  # noqa: E402

ABSTAIN = "I don't know"
_ABS_MARKERS = ("don't know", "dont know", "do not know", "not sure", "unknown", "cannot")


def is_abstain(text: str) -> bool:
    t = text.strip().lower()
    return (not t) or any(m in t for m in _ABS_MARKERS)


# ── dense retriever for RAG ──────────────────────────────────────────────────
class Retriever:
    def __init__(self, passages: List[str], model_name="Qwen/Qwen3-Embedding-0.6B"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model_name)
        self.passages = passages
        self.emb = self.m.encode(passages, normalize_embeddings=True) if passages else np.zeros((0, 1))

    def top_k(self, query: str, k: int = 3) -> List[str]:
        if len(self.passages) == 0:
            return []
        q = self.m.encode([query], normalize_embeddings=True)[0]
        sims = self.emb @ q
        idx = np.argsort(sims)[::-1][:k]
        return [self.passages[int(i)] for i in idx]


# ── generation ───────────────────────────────────────────────────────────────
def _chat(tok, user_msg: str) -> str:
    """Wrap a user message in Qwen3's chat template (thinking OFF for short answers)."""
    try:
        return tok.apply_chat_template([{"role": "user", "content": user_msg}],
                                       add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)
    except TypeError:                                     # older template: no enable_thinking kw
        return tok.apply_chat_template([{"role": "user", "content": user_msg}],
                                       add_generation_prompt=True, tokenize=False)


def _gen(model, tok, prompt, max_tokens=12, constraint=None):
    from mlx_lm import generate
    kw = dict(max_tokens=max_tokens, verbose=False)
    if constraint is not None:
        kw["logits_processors"] = [constraint]
    out = generate(model, tok, prompt=prompt, **kw)
    return out.replace("<think>", " ").replace("</think>", " ").strip().split("\n")[0].strip()


def closed_book(model, tok, question) -> str:
    msg = ("Answer with just the name or value, in a few words. If you do not know, "
           f"reply exactly \"I don't know\".\nQuestion: {question}")
    return _gen(model, tok, _chat(tok, msg))


def rag(model, tok, question, retriever: Retriever, k=3) -> str:
    ctx = "\n".join(retriever.top_k(question, k))
    msg = ("Answer using ONLY the context, with just the name or value. If the context "
           f"does not contain the answer, reply exactly \"I don't know\".\n"
           f"Context:\n{ctx}\nQuestion: {question}")
    return _gen(model, tok, _chat(tok, msg))


def gate(model, tok, question, subj, prop, store: LocalStore) -> str:
    fillers = store.grounded_fillers(subj, prop)
    if not fillers:
        return ABSTAIN                                    # entity/role absent -> structural abstain
    prompt = _chat(tok, f"Answer with just the name or value.\nQuestion: {question}")
    seqs, _ = build_allowed(tok, prompt, fillers, refusal=ABSTAIN)
    con = GroundedConstraint(seqs, tok.eos_token_id)
    return _gen(model, tok, prompt, constraint=con)


def build_store(kb_items) -> LocalStore:
    return LocalStore().inject([{"entity": it.subj, it.prop: it.obj} for it in kb_items])
