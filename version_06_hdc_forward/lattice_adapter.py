"""
lattice_adapter.py — Ada as an in-loop memory adapter on a STOCK open model.
============================================================================
The killer property: the base model's pretraining is irrelevant to Ada-groundedness.
We wrap the model's generation loop with a logits processor that, at every step,
masks the vocabulary to ONLY the token continuations of Ada's grounded fillers for
the current (entity, role) — or a refusal token. By construction the model CANNOT
emit an answer that contradicts Ada; it must return Ada's value or refuse. This is
the design doc's "check at logit time, every step, before commit… the deterministic
guarantee lives in the wrapper, not the weights."

  weights   : any HF/MLX causal LM (here Qwen3) — a fluent proposal engine.
  adapter   : this file — the disposal law, driven by the HDC lattice (memory.py).

No fork of the model source: `mlx_lm.generate(..., logits_processors=[GroundedConstraint(...)])`.

This module is tokenizer-agnostic and MLX-optional so the masking logic is unit-
tested on CPU (see __main__); run_qwen.py drives the real Qwen3.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def _to_list(tokens) -> list:
    """tokens may be a python list, np.ndarray, or mx.array."""
    try:
        return list(tokens.tolist())
    except AttributeError:
        return list(tokens)


class GroundedConstraint:
    """A logits processor that confines generation to a set of allowed token
    sequences (the grounded fillers + refusal). Prefix-automaton over those seqs."""

    def __init__(self, allowed_seqs: Sequence[Sequence[int]], eos_id: int, vocab_size=None):
        self.allowed = sorted({tuple(int(t) for t in s) for s in allowed_seqs if len(s) > 0})
        self.eos_id = int(eos_id)
        self.V = int(vocab_size) if vocab_size is not None else None  # else inferred from logits
        self.baseline = None          # length of the prompt (set on first call)

    def allowed_next(self, gen: tuple) -> set:
        nxt, done = set(), False
        for seq in self.allowed:
            if seq[:len(gen)] == gen:
                if len(seq) > len(gen):
                    nxt.add(seq[len(gen)])
                else:
                    done = True       # a full grounded filler is complete here
        if done:
            nxt.add(self.eos_id)      # allow stopping once a filler is finished
        return nxt

    def __call__(self, tokens, logits):
        toks = _to_list(tokens)
        if self.baseline is None:
            self.baseline = len(toks)
        gen = tuple(toks[self.baseline:])
        nxt = self.allowed_next(gen)
        if not nxt:                   # nothing grounded reachable -> force stop
            nxt = {self.eos_id}
        return self._mask(logits, nxt)

    def _mask(self, logits, allowed: set):
        V = int(logits.shape[-1])     # infer from the model's actual output width
        allow = np.zeros(V, dtype=bool)
        allow[np.fromiter(allowed, dtype=np.int64, count=len(allowed))] = True
        try:
            import mlx.core as mx      # real run
            if isinstance(logits, mx.array):
                shaped = mx.array(allow).reshape((1,) * (logits.ndim - 1) + (V,))
                return mx.where(shaped, logits, mx.array(-1e9, dtype=logits.dtype))
        except ImportError:
            pass
        out = np.array(logits, dtype=np.float64)
        neg = np.full_like(out, -1e9)
        mask = allow.reshape((1,) * (out.ndim - 1) + (V,)) if out.ndim > 1 else allow
        return np.where(mask, out, neg)


def continuation_ids(tokenizer, prompt: str, filler: str) -> List[int]:
    """The exact token ids the model must emit to append `filler` after `prompt`
    (BPE-boundary safe: diff the two encodings; fall back to encoding ' '+filler)."""
    p = tokenizer.encode(prompt)
    full = tokenizer.encode(prompt + " " + filler)
    if full[:len(p)] == list(p):
        return list(full[len(p):])
    return list(tokenizer.encode(" " + filler))


def build_allowed(tokenizer, prompt: str, fillers: Sequence[str], refusal: str) -> tuple:
    """Grounded filler continuations (+ refusal if the grounded set is empty)."""
    seqs = [continuation_ids(tokenizer, prompt, f) for f in fillers]
    refusal_seq = continuation_ids(tokenizer, prompt, refusal)
    if not seqs:
        seqs = [refusal_seq]
    return seqs, refusal_seq


# ── CPU unit test of the masking logic (no model, no mlx) ────────────────────
class _StubTok:
    """word-level tokenizer: token id = index in a fixed vocab."""
    def __init__(self, words): self.stoi = {w: i for i, w in enumerate(words)}
    def encode(self, s): return [self.stoi[w] for w in s.split()]


if __name__ == "__main__":
    vocab = ["Erin", "lives", "in", "boston", "austin", "maine", "<eos>", "unknown", "the"]
    tok = _StubTok(vocab)
    V, eos = len(vocab), vocab.index("<eos>")
    prompt = "Erin lives in"

    def run(fillers, refusal="unknown", greedy_pref=None):
        seqs, _ = build_allowed(tok, prompt, fillers, refusal)
        c = GroundedConstraint(seqs, eos, V)
        toks = tok.encode(prompt)
        out = []
        for _ in range(4):
            logits = np.zeros(V)                       # uniform "model" proposal
            if greedy_pref:                            # bias toward a would-be hallucination
                logits[tok.stoi[greedy_pref]] = 10.0
            masked = c(np.array(toks), logits)
            nxt = int(np.argmax(masked))
            if nxt == eos:
                break
            out.append(vocab[nxt]); toks.append(nxt)
        return " ".join(out)

    # 1) model WANTS to hallucinate 'austin' but only 'boston' is grounded -> forced to boston
    print("grounded={boston}, model biased to 'austin' ->", run(["boston"], greedy_pref="austin"))
    assert run(["boston"], greedy_pref="austin") == "boston"
    # 2) multiple grounded fillers -> model may pick either (here uniform -> first allowed)
    print("grounded={boston,austin}                  ->", run(["boston", "austin"]))
    assert run(["boston", "austin"]) in ("boston", "austin")
    # 3) empty grounded set -> refusal
    print("grounded={} (fact absent)                 ->", run([]))
    assert run([]) == "unknown"
    print("\nlattice_adapter CPU logic test PASSED — model cannot leave the grounded set")
