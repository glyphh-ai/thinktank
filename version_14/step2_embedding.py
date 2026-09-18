"""
Step 2a — EMBEDDINGS
====================
Turn the (B, T) grid of integer ids into a (B, T, C) grid of vectors, and add
position information.

  token_emb : (vocab_size, C)   LEARNABLE, one vector per word  -> "what"
  pi_pos    : (T, C)            FIXED, π-freq sinusoidal        -> "where"

Forward:  x_ids (B,T) -> tok[x_ids] (B,T,C)  +  pi_pos[0:T] (T,C)  ->  (B,T,C)

Run:  python step2_embedding.py
"""

import numpy as np
from step1_data import get_batch, vocab_size, BLOCK_SIZE, itos

# ---- Hyperparameters ---------------------------------------------------------
C = 128  # embedding dimension ("n_embd"): how many numbers represent each token

# ---- π-frequency positional encoding ----------------------------------------
def pi_pos_encoding(T, C):
    pos = np.arange(T)[:, None].astype(np.float64)
    dim = np.arange(C)[None, :].astype(np.float64)
    freq = 1.0 / (np.pi ** (2.0 * (dim // 2) / C))
    enc = np.where(dim % 2 == 0, np.sin(pos * freq), np.cos(pos * freq))
    return enc

# ---- Parameters (LEARNABLE) --------------------------------------------------
def init_params(rng: np.random.Generator) -> dict:
    return {
        "tok_emb": rng.normal(0, 0.02, size=(vocab_size, C)),
    }

# ---- Forward -----------------------------------------------------------------
def embed(params: dict, x_ids: np.ndarray) -> np.ndarray:
    """
    x_ids: (B, T) integer ids
    returns: (B, T, C) float vectors = token meaning + position
    """
    B, T = x_ids.shape
    tok = params["tok_emb"][x_ids]            # (B, T, C)
    pos = pi_pos_encoding(T, C)               # (T, C) — deterministic
    return tok + pos                          # (B, T, C)

# ---- Demo --------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    rng = np.random.default_rng(7)
    params = init_params(rng)
    print("LEARNABLE PARAMETER SHAPES:")
    for name, p in params.items():
        print(f"  {name:8s}: {p.shape}  ({p.size} numbers)")
    print(f"\nπ-pos encoding: ({BLOCK_SIZE}, {C})  (fixed, not learned)")
    print()

    batch_rng = np.random.default_rng(42)
    x, _ = get_batch("train", batch_rng)
    print(f"input  x      : {x.shape}  (grid of ids)")

    e = embed(params, x)
    print(f"output embed  : {e.shape}  (grid of vectors)   <- gained the C axis")
    print()

    b, t = 0, 0
    the_id = int(x[b, t])
    pos = pi_pos_encoding(x.shape[1], C)
    print(f"token at [b=0, t=0] is id {the_id} = {itos[the_id]!r}, sitting at position {t}")
    print("  token vector (first 6):", params["tok_emb"][the_id][:6])
    print("  π-pos vector (first 6):", pos[t][:6])
    print("  sum  = e[0,0] (first 6):", e[b, t][:6])
    print()
    print("Same word at a DIFFERENT position gets a DIFFERENT final vector,")
    print("because a different π-pos vector is added -> that's how order is encoded.")
