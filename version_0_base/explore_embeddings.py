"""
A guided tour of the embedding layer, one level at a time.
Run:  python explore_embeddings.py
Each section prints a small, concrete demonstration of one idea.
"""

import numpy as np
from step1_data import vocab_size, itos, stoi

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(7)

C = 8  # small embedding dim so whole vectors fit on screen

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# ---------------------------------------------------------------------------
section("LEVEL 1 — a raw id is just a meaningless label")
# The id 21 ('s') is NOT '21 units' of anything. Arithmetic on ids is nonsense.
print("id of 's' =", stoi["s"], " | id of 'b' =", stoi["b"], " | id of 'e' =", stoi["e"])
print("is 's' (21) '3 more than' 'p' (18)?  No -- ids are arbitrary name tags.")
print("We need each token as a POINT IN SPACE where math is meaningful.")

# ---------------------------------------------------------------------------
section("LEVEL 2 — the embedding TABLE: one learnable vector per token")
tok_emb = rng.normal(0, 0.02, size=(vocab_size, C))
print("tok_emb shape (vocab_size, C):", tok_emb.shape)
print("row i  = the vector for token id i")
print("col j  = feature j (a dimension the model defines for itself)")
print("\nrow for 's' (id 21):", tok_emb[stoi['s']])
print("row for 'e' (id  8):", tok_emb[stoi['e']])

# ---------------------------------------------------------------------------
section("LEVEL 3 — the lookup IS a one-hot matrix multiply (just faster)")
# one-hot of 's': a length-26 vector, 1 at index 21, 0 elsewhere.
onehot_s = np.zeros(vocab_size)
onehot_s[stoi["s"]] = 1.0
print("one-hot('s') (length 26):")
print(onehot_s)
print("\none-hot @ tok_emb  selects exactly row 21:")
print("   via matmul :", onehot_s @ tok_emb)
print("   via lookup :", tok_emb[stoi["s"]])
print("   identical? ", np.allclose(onehot_s @ tok_emb, tok_emb[stoi["s"]]))
print("=> 'embedding lookup' = multiply a one-hot by the table. Lookup skips the zeros.")

# ---------------------------------------------------------------------------
section("LEVEL 4 — why init RANDOM and SMALL (not zeros)?")
zeros_tbl = np.zeros((vocab_size, C))
print("If the table were all zeros, every token -> same vector:")
print("   'a' ->", zeros_tbl[stoi['a']], " 'z'... ->", zeros_tbl[stoi['y']])
print("   The model could never tell tokens apart, and (we'll see in backprop)")
print("   identical rows get identical gradients -> they'd stay identical forever.")
print("   RANDOM init breaks this symmetry so each token can specialize.")
print("SMALL (std=0.02): keeps early activations tiny & stable so nothing explodes.")
print("   sample std of our init:", round(tok_emb.std(), 4), "(~0.02)")

# ---------------------------------------------------------------------------
section("LEVEL 5 — fancy indexing turns a GRID of ids into a GRID of vectors")
x = np.array([[stoi[c] for c in "to be"],
              [stoi[c] for c in "or to"]])      # (B=2, T=5)
print("x ids (B=2, T=5):")
print(x)
e = tok_emb[x]                                   # (B, T, C)
print("\ntok_emb[x] shape:", e.shape, " <- gained the C axis")
print("e[0,0] (vector for x[0,0]=%d '%s'):" % (x[0,0], itos[x[0,0]]))
print("   ", e[0,0])
print("same as tok_emb[%d]? " % x[0,0], np.allclose(e[0,0], tok_emb[x[0,0]]))

# ---------------------------------------------------------------------------
section("LEVEL 6 — positions: arange + a SECOND table")
BLOCK = 16
pos_emb = rng.normal(0, 0.02, size=(BLOCK, C))
T = x.shape[1]
print("np.arange(T) =", np.arange(T), " (the slot numbers 0..T-1)")
pos = pos_emb[np.arange(T)]                       # (T, C)
print("pos = pos_emb[arange(T)] shape:", pos.shape, "(one vector per position)")
print("pos[0] (position-0 vector):", pos[0])

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
section("LEVEL 8 — same char, different position -> different final vector")
# 'o' appears in both rows but at different slots.
print("'o' id =", stoi['o'])
# row0 'to be': 'o' at position 1 ; row1 'or to': 'o' at position 0
print("'o' at position 1 -> final:", (tok_emb[stoi['o']] + pos[1]))
print("'o' at position 0 -> final:", (tok_emb[stoi['o']] + pos[0]))
print("Same token vector, DIFFERENT position vector added => order is encoded.")
