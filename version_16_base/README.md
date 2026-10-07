# version_16_base — a GPT that speaks by step 5,000

A new from-scratch base. It is still NumPy with every gradient written by hand,
like every earlier version, but it is a real small GPT instead of the
1-block / 1-head substrate of `version_0_base`.

| | version_0_base | **version_16_base** |
|---|---|---|
| blocks × heads | 1 × 1 | **4 × 4** |
| width C / context T | 128 / 32 | **256 / 64** |
| params | 1.58M | **5.02M** |
| activation / head | ReLU / separate | **GELU / tied to tok_emb** |
| optimizer | Adam, fixed lr | **AdamW (wd 0.1, β2 0.95), warmup 200 + cosine, clip 1.0** |
| dtype | float64 | **float32** |
| data | TinyStories-valid, 90/10 split (4.5M words) | **train: first 100 MB of TinyStories-train (24M words); val: TinyStories-valid (4.5M words, never trained on)** |
| vocab | 5,375 (freq ≥ 10) | **7,218 (freq ≥ 20, 0.25% `<unk>`)** |
| steps | 8,000 | **5,000** (10.2M tokens, < 1 epoch) |

## Files
| file | role |
|---|---|
| `step1_data.py` | tokenizer, vocab, cached encoded corpus, `get_batch` |
| `step3_gpt.py` | model, manual backprop, AdamW, schedule, train, sample |
| `test_train.py` | finite-difference gradient check on every tensor + it-learns check |
| `strat0_train.py` | the 5,000-step base run → `strat0_params.npz` |
| `sample.py` | generate from a checkpoint, optional prompt |

## Reproduce
```bash
mkdir -p data
curl -L -o data/tinystories_valid.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt
curl -L -r 0-104857599 -o data/tinystories_train_100mb.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt
python test_train.py        # ~2 min
python strat0_train.py      # ~1 h, 4-core CPU (~0.65 s/step)
python sample.py "once upon a time, a little dog"
```

Results: see `FINDINGS.md`.
