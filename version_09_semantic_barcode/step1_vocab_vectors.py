"""step1 — extract concept vectors for the 1027-word eval vocabulary.

Same measurement protocol as v07 step1 (12 carrier frames, residual stream at
the word's last subword token, every layer, split-half + usage/mention
means), applied to wordlist.txt instead of the v07 inventory. ~12.3k forward
passes on CPU. Output: vocab_vectors.npz (gitignored, ~350 MB).
"""
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "version_07_prime_basis"))
from concepts import FRAME_KIND, TEMPLATES  # noqa: E402
from step1_vectors import template_char_span  # noqa: E402

from vocab import load_vocab  # noqa: E402

SPLITS = ["half_a", "half_b", "usage", "mention"]
OUT = os.path.join(HERE, "vocab_vectors.npz")


def main(model_name="Qwen/Qwen2.5-0.5B", batch_size=32):
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    words = load_vocab()
    items = []
    for wi, w in enumerate(words):
        for ti, t in enumerate(TEMPLATES):
            splits = [0 if ti % 2 == 0 else 1, 2 if FRAME_KIND[ti] == "usage" else 3]
            items.append((wi, splits, t.replace("{w}", w), template_char_span(t, w)))

    n_layers = model.config.num_hidden_layers + 1
    d = model.config.hidden_size
    sums = np.zeros((len(words), len(SPLITS), n_layers, d), dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, len(items), batch_size):
            batch = items[b0:b0 + batch_size]
            enc = tok([it[2] for it in batch], return_tensors="pt", padding=True,
                      return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")
            hs = torch.stack(model(**enc, output_hidden_states=True).hidden_states, dim=1)
            for i, (wi, spl, _text, (cs, ce)) in enumerate(batch):
                tok_idx = None
                for j in range(offsets.shape[1]):
                    s, e = offsets[i, j].tolist()
                    if e > s and s < ce and e > cs:
                        tok_idx = j
                assert tok_idx is not None, f"no token for span in {_text!r}"
                vec = hs[i, :, tok_idx, :].numpy()
                for sp in spl:
                    sums[wi, sp] += vec
            if (b0 // batch_size) % 40 == 0:
                print(f"  {b0 + len(batch)}/{len(items)} passes, {time.time()-t0:.0f}s", flush=True)

    counts = np.array([len(TEMPLATES) // 2, len(TEMPLATES) // 2,
                       FRAME_KIND.count("usage"), FRAME_KIND.count("mention")], dtype=np.int32)
    np.savez_compressed(OUT, sums=sums, counts=counts, words=np.array(words),
                        split_names=np.array(SPLITS), model=model_name,
                        n_layers=n_layers, d=d)
    print(f"wrote {OUT} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
