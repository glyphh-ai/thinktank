"""step1 — extract activation-space concept vectors from a tiny stock LM.

For every word w (61 primes + 130 control pool + 100 targets) and each of 12
carrier templates, run one forward pass and read the residual stream at the
LAST subword token of w, at every layer. Per word we keep two vectors per
layer — the mean over EVEN-indexed templates (half A) and over ODD-indexed
templates (half B) — plus usage-only and mention-only means for the frame
confound check.

Post-processing per layer (done downstream, but the raw sums are stored here):
  center  — subtract the grand mean over all words (removes the carrier frame)
  normalize — unit norm

Output: vectors.npz
  sums   float32 [n_words, n_splits=4, n_layers, d]   (A, B, usage, mention)
  counts int32   [n_splits]
  words, split_names, model metadata

Runs on CPU. ~3.5k forward passes of ~15-token sequences.
"""
import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from concepts import FRAME_KIND, TEMPLATES, all_words

SPLITS = ["half_a", "half_b", "usage", "mention"]


def template_char_span(template: str, word: str):
    start = template.index("{w}")
    return start, start + len(word)


def build_batch_items():
    """(word_idx, split memberships, filled text, char span) for every (word, template)."""
    items = []
    words = all_words()
    for wi, w in enumerate(words):
        for ti, t in enumerate(TEMPLATES):
            filled = t.replace("{w}", w)
            cs, ce = template_char_span(t, w)
            splits = [0 if ti % 2 == 0 else 1, 2 if FRAME_KIND[ti] == "usage" else 3]
            items.append((wi, splits, filled, (cs, ce)))
    return words, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="vectors.npz")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    words, items = build_batch_items()
    n_layers = model.config.num_hidden_layers + 1  # + embedding layer
    d = model.config.hidden_size
    sums = np.zeros((len(words), len(SPLITS), n_layers, d), dtype=np.float32)
    counts = np.zeros(len(SPLITS), dtype=np.int32)  # per word, identical across words

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # keep offsets aligned; pad tokens map to (0,0)

    t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, len(items), args.batch_size):
            batch = items[b0:b0 + args.batch_size]
            texts = [it[2] for it in batch]
            enc = tok(texts, return_tensors="pt", padding=True,
                      return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")
            out = model(**enc, output_hidden_states=True)
            # hidden_states: tuple of [B, T, d], one per layer incl. embeddings
            hs = torch.stack(out.hidden_states, dim=1)  # [B, n_layers, T, d]
            for i, (wi, splits, _text, (cs, ce)) in enumerate(batch):
                # last non-special token overlapping the word's char span
                tok_idx = None
                for j in range(offsets.shape[1]):
                    s, e = offsets[i, j].tolist()
                    if e > s and s < ce and e > cs:
                        tok_idx = j
                assert tok_idx is not None, f"no token found for span in: {_text!r}"
                vec = hs[i, :, tok_idx, :].numpy()
                for sp in splits:
                    sums[wi, sp] += vec
            if b0 == 0:
                print(f"first batch ok ({len(batch)} seqs, {hs.shape[2]} tokens max)")
            if (b0 // args.batch_size) % 20 == 0:
                done = b0 + len(batch)
                print(f"  {done}/{len(items)} passes, {time.time()-t0:.0f}s", flush=True)

    n_templates = len(TEMPLATES)
    counts[:] = [n_templates // 2, n_templates // 2,
                 FRAME_KIND.count("usage"), FRAME_KIND.count("mention")]

    np.savez_compressed(
        args.out, sums=sums, counts=counts,
        words=np.array(words), split_names=np.array(SPLITS),
        model=args.model, n_layers=n_layers, d=d)
    print(f"wrote {args.out}: {len(words)} words x {len(SPLITS)} splits x "
          f"{n_layers} layers x {d} dims in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
