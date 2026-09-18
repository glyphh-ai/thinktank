"""
Side-by-side generation: v0 baseline (learned pos) vs v14 3-phase π-pos.
Same seeds, same temperature, same prompt — only the position mechanism differs.

Run:  python compare_generation.py
"""

import numpy as np
import sys
sys.path.insert(0, "../version_0_base")
import step3_gpt as G_baseline
sys.path.pop(0)

from step1_data import vocab_size, BLOCK_SIZE, decode, stoi
from probe_phases import multi_phase_pi_pos, forward_with_attention

C = 128
BLOCK = BLOCK_SIZE
EPS = 1e-5


def sample_baseline(p, n=200, seed=42, temperature=0.8):
    rng = np.random.default_rng(seed)
    out = [stoi["<eot>"]]
    for _ in range(n):
        x = np.array([out[-BLOCK:]])
        logits, _ = G_baseline.forward(p, x)
        z = logits[0, -1] / temperature
        probs = np.exp(z - z.max())
        probs /= probs.sum()
        nxt = int(rng.choice(len(probs), p=probs))
        out.append(nxt)
    return decode(out[1:])


def sample_3phase(p, n=200, seed=42, temperature=0.8):
    rng = np.random.default_rng(seed)
    out = [stoi["<eot>"]]
    for _ in range(n):
        x = np.array([out[-BLOCK:]])
        logits, _ = forward_with_attention(p, x, n_phases=3)
        z = logits[0, -1] / temperature
        probs = np.exp(z - z.max())
        probs /= probs.sum()
        nxt = int(rng.choice(len(probs), p=probs))
        out.append(nxt)
    return decode(out[1:])


if __name__ == "__main__":
    # load both
    base_path = "../version_0_base/strat0_params.npz"
    phase_path = "strat0_params_3phase.npz"

    print("Loading baseline (v0, learned pos)...")
    bl = np.load(base_path)
    p_base = {k: bl[k] for k in bl.files}

    print("Loading 3-phase (v14, π-pos)...")
    pl = np.load(phase_path)
    p_3phase = {k: pl[k] for k in pl.files}

    for seed in [42, 7, 123]:
        for temp in [0.7, 0.9]:
            print(f"\n{'='*70}")
            print(f"seed={seed}  temperature={temp}")
            print(f"{'='*70}")

            print(f"\n--- BASELINE (learned pos, {sum(v.size for v in p_base.values()):,} params) ---")
            print(sample_baseline(p_base, n=150, seed=seed, temperature=temp))

            print(f"\n--- 3-PHASE π-pos ({sum(v.size for v in p_3phase.values()):,} params) ---")
            print(sample_3phase(p_3phase, n=150, seed=seed, temperature=temp))
