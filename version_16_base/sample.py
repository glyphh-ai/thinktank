"""
Generate from a trained checkpoint.

Run:  python sample.py                                   # free story
      python sample.py "once upon a time, a little dog"  # continue a prompt
      python sample.py --n 300 --temperature 0.9 --seed 3 "lily went to the park"
"""

import argparse

import numpy as np

import step3_gpt as G
from step1_data import encode

ap = argparse.ArgumentParser()
ap.add_argument("prompt", nargs="?", default=None)
ap.add_argument("--ckpt", default="strat0_params.npz")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top_k", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

ck = np.load(args.ckpt)
p = {k: ck[k] for k in ck.files if k != "step"}
print(f"[checkpoint step {int(ck['step'])}]\n")
print(G.sample(p, n=args.n, seed=args.seed, temperature=args.temperature,
               top_k=args.top_k, prompt_ids=encode(args.prompt) if args.prompt else None))
