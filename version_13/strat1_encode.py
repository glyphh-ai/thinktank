"""
Strat 1 — COMPILE: readings.jsonl -> the lattice vector bank on disk
====================================================================
Turns the judged readings into reusable artifacts, aligned to vocab id
order so ANY consumer (version_0_base included — same corpus, same vocab)
can index straight into them with token ids:

    lattice_bank.npz
        cortex      (V, D) int8 ternary  — one cortex vector per word
        judged      (V,)   bool          — row is real (False = zero row:
                                           <unk> or missing reading)
        vocab_sha   str                  — must equal the consumer's vocab
                                           fingerprint or REFUSE to load

Load from anywhere:

    from strat1_encode import load_bank
    bank = load_bank()          # verifies vocab_sha against step1_data

Run:  python strat1_encode.py           # after strat1_judge.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from step1_data import words as VOCAB, stoi
from strat1_schema import ROLE_PATHS
from strat1_vsa import D, encode_word

HERE = Path(__file__).parent
READINGS = HERE / "readings.jsonl"
BANK = HERE / "lattice_bank.npz"


def vocab_fingerprint(vocab) -> str:
    return hashlib.sha256("\n".join(vocab).encode()).hexdigest()[:16]


def compile_bank():
    readings = {}
    with READINGS.open() as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                readings[rec["word"]] = rec["reading"]

    V = len(VOCAB)
    cortex = np.zeros((V, D), dtype=np.int8)
    judged = np.zeros(V, dtype=bool)
    for word, reading in readings.items():
        i = stoi.get(word)
        if i is None:
            continue  # reading from a stale vocab — fingerprint will differ anyway
        cortex[i] = encode_word(reading)["cortex"]
        judged[i] = True

    np.savez_compressed(
        BANK, cortex=cortex, judged=judged,
        vocab_sha=np.array(vocab_fingerprint(VOCAB)),
    )
    print(f"words judged : {judged.sum()}/{V}")
    print(f"vocab_sha    : {vocab_fingerprint(VOCAB)}")
    print(f"saved        : {BANK.name}  "
          f"({BANK.stat().st_size / 1e6:.0f} MB compressed)")


def load_bank(path: Path = BANK) -> dict:
    """Load the bank; refuse if it was built for a different vocab."""
    z = np.load(path, allow_pickle=False)
    sha = str(z["vocab_sha"])
    want = vocab_fingerprint(VOCAB)
    if sha != want:
        raise ValueError(
            f"lattice bank vocab_sha {sha} != current vocab {want} — "
            "the bank was built for a different corpus/vocab; re-run "
            "strat1_judge.py + strat1_encode.py")
    return {"cortex": z["cortex"], "judged": z["judged"], "vocab_sha": sha}


if __name__ == "__main__":
    compile_bank()
