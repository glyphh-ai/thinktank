"""
Make the 'point in space' idea literal: a C=2 embedding for a sample of words,
plotted as labeled dots. With C=2 we can SEE the coordinates directly.
Run:  python plot_embedding_space.py   (writes embedding_space.png)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from step1_data import words, vocab_size, stoi

C = 2  # two dims so each token is literally an (x, y) point we can plot
rng = np.random.default_rng(7)
tok_emb = rng.normal(0, 0.02, size=(vocab_size, C))

SAMPLE_WORDS = [
    "the", "a", "cat", "dog", "big", "small", "happy", "sad",
    "run", "walk", "said", "saw", "she", "he", "once", "very",
    "good", "bad", "little", "story", "love", "time", "day", "night",
]
sample_ids = [stoi[w] for w in SAMPLE_WORDS if w in stoi]

print("With C=2, each row is an (x, y) coordinate:")
for w in SAMPLE_WORDS[:5]:
    i = stoi[w]
    print(f"  {w!r:10s} id {i:4d} -> (x={tok_emb[i,0]:+.4f}, y={tok_emb[i,1]:+.4f})")

# --- plot ---
plt.figure(figsize=(8, 8))
plt.scatter(tok_emb[:, 0], tok_emb[:, 1], s=4, color="lightgray", alpha=0.3,
            label=f"all {vocab_size} words")
plt.scatter(tok_emb[sample_ids, 0], tok_emb[sample_ids, 1], s=30, color="steelblue")
for idx in sample_ids:
    plt.annotate(words[idx], (tok_emb[idx, 0], tok_emb[idx, 1]),
                 fontsize=9, ha="left", va="bottom")
plt.axhline(0, color="#ddd", lw=0.8); plt.axvline(0, color="#ddd", lw=0.8)
plt.title(f"Token embedding space at INIT (C=2, {vocab_size} words)\n"
          "random now -> training moves the dots")
plt.xlabel("feature 0  (axis 0)"); plt.ylabel("feature 1  (axis 1)")
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("embedding_space.png", dpi=110)
print("\nsaved embedding_space.png")

# --- geometry is what matters, and it works at ANY dimension ---
def dist(a, b): return float(np.linalg.norm(a - b))
print("\nDistances between points (this is the 'meaning' signal):")
print(f"  'cat'<->'dog':   {dist(tok_emb[stoi['cat']], tok_emb[stoi['dog']]):.4f}")
print(f"  'cat'<->'happy': {dist(tok_emb[stoi['cat']], tok_emb[stoi['happy']]):.4f}")
print("(random now, so distances are meaningless -- after training, similar")
print(" words would sit CLOSE together; that closeness is what the model uses.)")
