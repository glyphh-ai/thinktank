"""
A guided tour of the embedding layer, one level at a time.
Run:  python explore_embeddings.py
Each section prints a small, concrete demonstration of one idea.
"""

import numpy as np
from step1_data import vocab_size, itos, stoi, words

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(7)

C = 8  # small embedding dim so whole vectors fit on screen

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# ---------------------------------------------------------------------------
section("LEVEL 1 — a raw id is just a meaningless label")
print("id of 'the' =", stoi["the"], " | id of 'cat' =", stoi["cat"],
      " | id of 'happy' =", stoi["happy"])
print("is 'the' 'more than' 'cat'?  No -- ids are arbitrary name tags.")
print("We need each token as a POINT IN SPACE where math is meaningful.")

# ---------------------------------------------------------------------------
section("LEVEL 2 — the embedding TABLE: one learnable vector per token")
tok_emb = rng.normal(0, 0.02, size=(vocab_size, C))
print("tok_emb shape (vocab_size, C):", tok_emb.shape)
print("row i  = the vector for token id i")
print("col j  = feature j (a dimension the model defines for itself)")
print("\nrow for 'the' (id %d):" % stoi['the'], tok_emb[stoi['the']])
print("row for 'cat' (id %d):" % stoi['cat'], tok_emb[stoi['cat']])

# ---------------------------------------------------------------------------
section("LEVEL 3 — the lookup IS a one-hot matrix multiply (just faster)")
onehot = np.zeros(vocab_size)
onehot[stoi["the"]] = 1.0
print(f"one-hot('the') (length {vocab_size}): mostly zeros, 1 at index {stoi['the']}")
print("\none-hot @ tok_emb  selects exactly that row:")
print("   via matmul :", (onehot @ tok_emb))
print("   via lookup :", tok_emb[stoi["the"]])
print("   identical? ", np.allclose(onehot @ tok_emb, tok_emb[stoi["the"]]))
print("=> 'embedding lookup' = multiply a one-hot by the table. Lookup skips the zeros.")

# ---------------------------------------------------------------------------
section("LEVEL 4 — why init RANDOM and SMALL (not zeros)?")
zeros_tbl = np.zeros((vocab_size, C))
print("If the table were all zeros, every token -> same vector:")
print("   'the' ->", zeros_tbl[stoi['the']], " 'cat' ->", zeros_tbl[stoi['cat']])
print("   The model could never tell tokens apart, and (we'll see in backprop)")
print("   identical rows get identical gradients -> they'd stay identical forever.")
print("   RANDOM init breaks this symmetry so each token can specialize.")
print("SMALL (std=0.02): keeps early activations tiny & stable so nothing explodes.")
print("   sample std of our init:", round(tok_emb.std(), 4), "(~0.02)")

# ---------------------------------------------------------------------------
section("LEVEL 5 — fancy indexing turns a GRID of ids into a GRID of vectors")
x = np.array([[stoi["the"], stoi["cat"], stoi["sat"]],
              [stoi["a"],   stoi["big"], stoi["dog"]]])   # (B=2, T=3)
print("x ids (B=2, T=3):")
print(x)
e = tok_emb[x]                                   # (B, T, C)
print("\ntok_emb[x] shape:", e.shape, " <- gained the C axis")
print("e[0,0] (vector for x[0,0]=%d '%s'):" % (x[0,0], itos[x[0,0]]))
print("   ", e[0,0])
print("same as tok_emb[%d]? " % x[0,0], np.allclose(e[0,0], tok_emb[x[0,0]]))

# ---------------------------------------------------------------------------
section("LEVEL 6 — positions: π-frequency sinusoidal encoding")
BLOCK = 16
T = x.shape[1]

def pi_pos_encoding(T, C):
    pos = np.arange(T)[:, None].astype(np.float64)
    dim = np.arange(C)[None, :].astype(np.float64)
    freq = 1.0 / (np.pi ** (2.0 * (dim // 2) / C))
    return np.where(dim % 2 == 0, np.sin(pos * freq), np.cos(pos * freq))

pos = pi_pos_encoding(T, C)                       # (T, C)
print("π-pos encoding shape:", pos.shape, "(one vector per position, deterministic)")
print("pos[0] (position-0 vector):", pos[0])
print("pos[1] (position-1 vector):", pos[1])
print("No learned parameters — frequencies are powers of π.")

# ---------------------------------------------------------------------------
section("LEVEL 7 — the ADD + broadcasting (exact shapes)")
print("tok lookup e :", e.shape, " (B, T, C)")
print("pos          :", pos.shape, "   (T, C)")
print("numpy aligns from the RIGHT:  (B,T,C) vs (1,T,C) -> broadcast over B")
out = e + pos
print("e + pos      :", out.shape, " (B, T, C)")
print("\nProof every sequence gets the SAME pos added:")
print("  e[0,0] + pos[0] == out[0,0]? ", np.allclose(e[0,0] + pos[0], out[0,0]))
print("  e[1,0] + pos[0] == out[1,0]? ", np.allclose(e[1,0] + pos[0], out[1,0]))

# ---------------------------------------------------------------------------
section("LEVEL 8 — same word, different position -> different final vector")
w = "the"
print(f"'{w}' id =", stoi[w])
vec = tok_emb[stoi[w]]
print(f"'{w}' at position 0 -> final:", (vec + pos[0]))
print(f"'{w}' at position 1 -> final:", (vec + pos[1]))
print("Same token vector, DIFFERENT π-pos vector added => order is encoded.")
