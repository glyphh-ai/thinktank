"""
Step 1 — DATA (word-level, real train/val split)
================================================
Goal: turn raw text into integer batches of (input, target) pairs.

Same word tokenizer as version_0_base, but more data and an honest split:
  train : first 100 MB of TinyStories-train   (~22M words)
  val   : TinyStories-valid                   (~4.5M words, never trained on)
Vocab: words with freq >= MIN_FREQ in train (~8k words, ~99.8% coverage).

The encoded ids are cached to data/*.npy, so only the first run pays for
tokenization (~1 min).

Get the data (once):
  curl -L -o data/tinystories_valid.txt \
    https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt
  curl -L -r 0-104857599 -o data/tinystories_train_100mb.txt \
    https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt

Run:  python step1_data.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
TRAIN_TXT = DATA_DIR / "tinystories_train_100mb.txt"
VAL_TXT = DATA_DIR / "tinystories_valid.txt"
MIN_FREQ = 20

# ---- 0. Tokenize -------------------------------------------------------------
# Lowercase words (contractions kept whole), punctuation as its own token,
# story boundary <|endoftext|> mapped to a single <eot> token.
TOKEN_RE = re.compile(r"<eot>|[a-z]+'[a-z]+|[a-z]+|[.,!?;:\"]")

def tokenize(text: str) -> list[str]:
    text = text.lower().replace("<|endoftext|>", " <eot> ")
    return TOKEN_RE.findall(text)

# ---- 1. Vocabulary + encoded corpus (cached) ---------------------------------
_cache = DATA_DIR / f"cache_f{MIN_FREQ}"

def _build():
    train_toks = tokenize(TRAIN_TXT.read_text())
    counts = Counter(train_toks)
    words = ["<unk>"] + sorted(w for w, n in counts.items() if n >= MIN_FREQ)
    stoi = {w: i for i, w in enumerate(words)}
    unk = stoi["<unk>"]
    enc = lambda toks: np.array([stoi.get(t, unk) for t in toks], dtype=np.int32)
    _cache.mkdir(exist_ok=True)
    np.save(_cache / "train.npy", enc(train_toks))
    np.save(_cache / "val.npy", enc(tokenize(VAL_TXT.read_text())))
    (_cache / "vocab.json").write_text(json.dumps(words))

if not (_cache / "vocab.json").exists():
    _build()

words = json.loads((_cache / "vocab.json").read_text())
vocab_size = len(words)
stoi = {w: i for i, w in enumerate(words)}
itos = {i: w for i, w in enumerate(words)}
UNK = stoi["<unk>"]
EOT = stoi["<eot>"]
train_data = np.load(_cache / "train.npy")
val_data = np.load(_cache / "val.npy")

def encode(toks) -> list[int]:
    """Text (or pre-tokenized list) -> list of token ids."""
    if isinstance(toks, str):
        toks = tokenize(toks)
    return [stoi.get(t, UNK) for t in toks]

def decode(ids) -> str:
    """List of token ids -> readable text (no space before punctuation)."""
    out = []
    for i in ids:
        w = itos[int(i)]
        if out and w in ".,!?;:":
            out[-1] += w
        elif w == "<eot>":
            out.append("\n\n")
        else:
            out.append(w)
    return " ".join(out).replace(" \n\n ", "\n\n")

# ---- 2. Batching -------------------------------------------------------------
BLOCK_SIZE = 64   # T: words of context

def get_batch(split: str, rng: np.random.Generator, B: int = 32, T: int = BLOCK_SIZE):
    """x, y each (B,T) int; y is x shifted by one -> next-word target."""
    src = train_data if split == "train" else val_data
    ix = rng.integers(0, len(src) - T - 1, size=B)
    x = np.stack([src[i:i + T] for i in ix])
    y = np.stack([src[i + 1:i + 1 + T] for i in ix])
    return x, y

# ---- Demo --------------------------------------------------------------------
if __name__ == "__main__":
    print(f"train tokens : {len(train_data):,}")
    print(f"val tokens   : {len(val_data):,}")
    print(f"vocab size   : {vocab_size}  (MIN_FREQ={MIN_FREQ})")
    print(f"<unk> rate   : train {(train_data == UNK).mean():.3%}  "
          f"val {(val_data == UNK).mean():.3%}")
    x, y = get_batch("train", np.random.default_rng(1337), B=2)
    print("\ninput  x[0]:", repr(decode(x[0])))
    print("target y[0]:", repr(decode(y[0])))
