"""
Step 1 — DATA
=============
Goal: turn raw text into integer batches of (input, target) pairs that the
GPT will learn from.

Run:  python step1_data.py
"""

import numpy as np

# A tiny corpus. Char-level modeling means we don't need much to see learning.
# (Swap this for a .txt file later; the rest of the pipeline won't change.)
TEXT = (
    "to be, or not to be, that is the question:\n"
    "whether tis nobler in the mind to suffer\n"
    "the slings and arrows of outrageous fortune,\n"
    "or to take arms against a sea of troubles\n"
    "and by opposing end them.\n"
) * 20  # repeat so there's enough data to batch

# ---- 1. Build the vocabulary -------------------------------------------------
# The set of every distinct character, sorted so the mapping is deterministic.
chars = sorted(set(TEXT))
vocab_size = len(chars)

# Two lookup tables: string->int (stoi) and int->string (itos).
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str) -> list[int]:
    """Text -> list of token ids."""
    return [stoi[c] for c in s]

def decode(ids: list[int]) -> str:
    """List of token ids -> text."""
    return "".join(itos[i] for i in ids)

# Encode the whole corpus once into a 1-D array of ids.
data = np.array(encode(TEXT), dtype=np.int64)

# ---- 2. Train / validation split --------------------------------------------
# Hold out the last 10% to check we're learning, not just memorizing.
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ---- 3. Batching -------------------------------------------------------------
BLOCK_SIZE = 16   # T: how many tokens of context the model sees at once
BATCH_SIZE = 4    # B: how many independent sequences per batch

def get_batch(split: str, rng: np.random.Generator):
    """
    Return one batch: x and y, each shape (BATCH_SIZE, BLOCK_SIZE), dtype int.
    y is x shifted by one position -> the "next char" target at every step.
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
    print(f"vocab size    : {vocab_size}")
    print(f"vocab         : {''.join(chars)!r}")
    print()

    print("encode('be') ->", encode("be"))
    print("decode(...)  ->", repr(decode(encode("be"))))
    print()

    rng = np.random.default_rng(1337)
    x, y = get_batch("train", rng)
    print(f"x shape: {x.shape}   y shape: {y.shape}")
    print("\nFirst sequence in the batch:")
    print("  input  x[0]:", x[0].tolist(), "->", repr(decode(x[0].tolist())))
    print("  target y[0]:", y[0].tolist(), "->", repr(decode(y[0].tolist())))
    print("\nRead it as 'given these chars, predict the next':")
    for t in range(6):
        context = x[0, : t + 1].tolist()
        target = y[0, t]
        print(f"  {repr(decode(context)):>22}  ->  {repr(itos[int(target)])}")
