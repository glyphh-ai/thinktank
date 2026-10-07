# version_17_ada — new design (starting point)

Copied unchanged from `version_13` as the starting point:

| file | role |
|---|---|
| `strat1_schema.py` | the lattice: layers > segments > roles, values from the 65 NSM primes |
| `strat1_vsa.py` | the encoder: bipolar atoms (D=100k), role⊙value bind, weighted bundle → one cortex vector per word; `unbind` reads a role back out |
| `strat1_judge.py` | fills each word's roles with claude-haiku-4-5 → `readings.jsonl` (needs `ANTHROPIC_API_KEY`) |
| `strat1_encode.py` | compiles `readings.jsonl` → `lattice_bank.npz` (one cortex per vocab word) |
| `step1_data.py` | the TinyStories-valid vocab the readings are locked to (vocab_sha `8a7c1e5989bbe53a`) |
| `readings.jsonl` | 5,374 judged words |

```bash
mkdir -p data && curl -L -o data/tinystories_valid.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt
python strat1_vsa.py      # self-test: encode, then decode every role back out
python strat1_encode.py   # -> lattice_bank.npz (~1 min, 84 MB, not committed)
```
