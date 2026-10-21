"""
Step 2a — EMBEDDINGS
====================
Turn the (B, T) grid of integer ids into a (B, T, C) grid of vectors, and add
position information.

Two LEARNABLE tables (the first parameters in our model):
  token_emb : (vocab_size, C)   one vector per word       -> "what" the token is
  pos_emb   : (block_size, C)   one vector per position    -> "where" it sits

Forward:  x_ids (B,T) -> tok[x_ids] (B,T,C)  +  pos[0:T] (T,C)  ->  (B,T,C)

Run:  python step2_embedding.py
"""

import numpy as np
from step1_data import get_batch, vocab_size, BLOCK_SIZE, itos

# ---- Hyperparameters ---------------------------------------------------------
C = 128  # embedding dimension ("n_embd"): how many numbers represent each token

# ---- Parameters (LEARNABLE) --------------------------------------------------
# Initialized to small random values. Adam will nudge these every step.
# A dict so the model's parameters are easy to grow and iterate over later.
def init_params(rng: np.random.Generator) -> dict:
    return {
        # one row per vocab id
        "tok_emb": rng.normal(0, 0.02, size=(vocab_size, C)),
        # one row per position 0..BLOCK_SIZE-1
        "pos_emb": rng.normal(0, 0.02, size=(BLOCK_SIZE, C)),
    }

# ---- Forward -----------------------------------------------------------------
def embed(params: dict, x_ids: np.ndarray) -> np.ndarray:
    """
    x_ids: (B, T) integer ids
    returns: (B, T, C) float vectors = token meaning + position
    """
    B, T = x_ids.shape

    # 1. Token lookup: index the table with EVERY id at once (fancy indexing).
    #    tok_emb[x_ids] has shape (B, T, C) -- each id replaced by its row.
    tok = params["tok_emb"][x_ids]            # (B, T, C)

    # 2. Position lookup: the first T position vectors, one per slot 0..T-1.
    pos = params["pos_emb"][np.arange(T)]     # (T, C)

    # 3. Add them. pos is (T, C); it BROADCASTS across the batch axis to (B,T,C),
    #    so every sequence gets the same position vectors added to it.
    return tok + pos                          # (B, T, C)

# ---- Demo --------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    rng = np.random.default_rng(7)
    params = init_params(rng)
    print("PARAMETER SHAPES (the things that will learn):")
    for name, p in params.items():
        print(f"  {name:8s}: {p.shape}  ({p.size} numbers)")
    print()

    batch_rng = np.random.default_rng(42)
    x, _ = get_batch("train", batch_rng)
    print(f"input  x      : {x.shape}  (grid of ids)")

    e = embed(params, x)
    print(f"output embed  : {e.shape}  (grid of vectors)   <- gained the C axis")
    print()

    # Peek at one token to see the add happen.
    b, t = 0, 0
    the_id = int(x[b, t])
    print(f"token at [b=0, t=0] is id {the_id} = {itos[the_id]!r}, sitting at position {t}")
    print("  token vector (first 6 of 32):", params["tok_emb"][the_id][:6])
    print("  pos   vector (first 6 of 32):", params["pos_emb"][t][:6])
    print("  sum  = e[0,0] (first 6 of 32):", e[b, t][:6])
    print()
    print("Same char id at a DIFFERENT position gets a DIFFERENT final vector,")
    print("because a different pos vector is added -> that's how order is encoded.")
