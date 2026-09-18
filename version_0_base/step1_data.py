"""
Step 1 — DATA (word-level)
==========================
Goal: turn raw text into integer batches of (input, target) pairs that the
GPT will learn from.

Word-level tokens on TinyStories validation split (~4.5M words).
Vocab: every word with freq >= 10 (~5.4k words, 99.6% coverage).

Run:  python step1_data.py
"""

import re
from collections import Counter
from pathlib import Path

import numpy as np

CORPUS_PATH = Path(__file__).parent / "data" / "tinystories_valid.txt"

# ---- 0. Tokenize -------------------------------------------------------------
# Lowercase words (contractions kept whole), punctuation as its own token,
# story boundary <|endoftext|> mapped to a single <eot> token.
TOKEN_RE = re.compile(r"<eot>|[a-z]+'[a-z]+|[a-z]+|[.,!?;:\"]")

def tokenize(text: str) -> list[str]:
    text = text.lower().replace("<|endoftext|>", " <eot> ")
    return TOKEN_RE.findall(text)

_tokens = tokenize(CORPUS_PATH.read_text())

# ---- 1. Build the vocabulary -------------------------------------------------
# Words seen >= MIN_FREQ times get an id; everything else becomes <unk>.
# At MIN_FREQ=10 that is ~5.4k words covering 99.57% of the corpus — small
# enough that every word can later be judged and encoded individually.
MIN_FREQ = 10
_counts = Counter(_tokens)
words = ["<unk>"] + sorted(w for w, n in _counts.items() if n >= MIN_FREQ)
vocab_size = len(words)

# Two lookup tables: string->int (stoi) and int->string (itos).
stoi = {w: i for i, w in enumerate(words)}
itos = {i: w for i, w in enumerate(words)}
UNK = stoi["<unk>"]

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
        else:
            out.append(w)
    return " ".join(out)

# Encode the whole corpus once into a 1-D array of ids.
data = np.array(encode(_tokens), dtype=np.int64)

# ---- 2. Train / validation split --------------------------------------------
# Hold out the last 10% to check we're learning, not just memorizing.
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ---- 3. Batching -------------------------------------------------------------
BLOCK_SIZE = 32   # T: how many words of context the model sees at once
BATCH_SIZE = 4    # B: how many independent sequences per batch

def get_batch(split: str, rng: np.random.Generator):
    """
    Return one batch: x and y, each shape (BATCH_SIZE, BLOCK_SIZE), dtype int.
    y is x shifted by one position -> the "next word" target at every step.
    """
    src = train_data if split == "train" else val_data
    # Random starting indices, each leaving room for BLOCK_SIZE+1 tokens.
    ix = rng.integers(0, len(src) - BLOCK_SIZE, size=BATCH_SIZE)
    x = np.stack([src[i : i + BLOCK_SIZE] for i in ix])
    y = np.stack([src[i + 1 : i + 1 + BLOCK_SIZE] for i in ix])
    return x, y

# ---- Demo --------------------------------------------------------------------
if __name__ == "__main__":
    print(f"corpus length : {len(data)} tokens")
    print(f"vocab size    : {vocab_size}  (MIN_FREQ={MIN_FREQ})")
    unk_rate = float((data == UNK).mean())
    print(f"<unk> rate    : {unk_rate:.4%}")
    print()

    print("encode('the cat sat.') ->", encode("the cat sat."))
    print("decode(...)            ->", repr(decode(encode("the cat sat."))))
    print()

    rng = np.random.default_rng(1337)
    x, y = get_batch("train", rng)
    print(f"x shape: {x.shape}   y shape: {y.shape}")
    print("\nFirst sequence in the batch:")
    print("  input  x[0]:", repr(decode(x[0].tolist())))
    print("  target y[0]:", repr(decode(y[0].tolist())))
