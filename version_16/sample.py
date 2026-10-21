"""
Sample — generate from a trained checkpoint
===========================================
Run:  python sample.py                          # open a fresh story
      python sample.py "once upon a time, a little dog"
"""

import sys

import torch

from data import decode, load_vocab, tokenize
from model import GPT, Config
from train import CKPT

if __name__ == "__main__":
    words, stoi = load_vocab()
    ck = torch.load(CKPT)
    model = GPT(Config(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    model.eval()

    prompt = sys.argv[1] if len(sys.argv) > 1 else ""
    ids = [stoi["<eot>"]] + [stoi.get(t, 0) for t in tokenize(prompt)]
    torch.manual_seed(0)
    for _ in range(3):
        out = model.generate(torch.tensor([ids]), 150)[0, 1:].tolist()
        text = decode(out, words).split("<eot>")[0]
        print(text.strip(), "\n")
