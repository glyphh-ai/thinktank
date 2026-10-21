"""
Data — word-level TinyStories -> token id files
===============================================
Same tokenizer as version_0_base (lowercase words, contractions whole,
punctuation as tokens, story boundary -> <eot>), but on a slice of the
TinyStories *train* split so a bigger model has enough to learn from.

Vocab: the VOCAB_SIZE-1 most frequent words + <unk>.
Out:   data/train.bin, data/val.bin (uint16 ids), data/vocab.json

Run:  python data.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
TRAIN_TXT = DATA / "tinystories_train.txt"
VALID_TXT = DATA / "tinystories_valid.txt"

TRAIN_CHARS = 400_000_000   # ~90M words of the 1.9GB train split
VOCAB_SIZE = 8192
CHUNK = 20_000_000

TOKEN_RE = re.compile(r"<eot>|[a-z]+'[a-z]+|[a-z]+|[.,!?;:\"]")


def tokenize(text):
    text = text.lower().replace("<|endoftext|>", " <eot> ")
    return TOKEN_RE.findall(text)


def chunks(path, limit=None):
    """Yield text in CHUNK-sized pieces, split on story boundaries."""
    read = 0
    buf = ""
    with open(path, encoding="utf-8") as f:
        while limit is None or read < limit:
            piece = f.read(CHUNK)
            if not piece:
                break
            read += len(piece)
            buf += piece
            cut = buf.rfind("<|endoftext|>")
            if cut == -1:
                continue
            yield buf[:cut + len("<|endoftext|>")]
            buf = buf[cut + len("<|endoftext|>"):]
    if buf and (limit is None or read < limit):
        yield buf


def load_vocab():
    words = json.loads((DATA / "vocab.json").read_text())
    return words, {w: i for i, w in enumerate(words)}


def decode(ids, words):
    out = []
    for i in ids:
        w = words[int(i)]
        if out and w in ".,!?;:":
            out[-1] += w
        else:
            out.append(w)
    return " ".join(out)


def encode_file(path, stoi, out_path, limit=None):
    unk = stoi["<unk>"]
    parts = []
    for text in chunks(path, limit):
        parts.append(np.array([stoi.get(t, unk) for t in tokenize(text)], dtype=np.uint16))
    ids = np.concatenate(parts)
    ids.tofile(out_path)
    return ids


if __name__ == "__main__":
    counts = Counter()
    for text in chunks(TRAIN_TXT, TRAIN_CHARS):
        counts.update(tokenize(text))
    words = ["<unk>"] + [w for w, _ in counts.most_common(VOCAB_SIZE - 1)]
    stoi = {w: i for i, w in enumerate(words)}
    (DATA / "vocab.json").write_text(json.dumps(words))

    tr = encode_file(TRAIN_TXT, stoi, DATA / "train.bin", TRAIN_CHARS)
    va = encode_file(VALID_TXT, stoi, DATA / "val.bin")
    print(f"vocab {len(words)} | train {len(tr):,} tokens "
          f"(unk {np.mean(tr == 0):.3%}) | val {len(va):,} tokens (unk {np.mean(va == 0):.3%})")
