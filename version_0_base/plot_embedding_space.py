"""
Make the 'point in space' idea literal: a C=2 embedding for all 26 chars,
plotted as labeled dots. With C=2 we can SEE the coordinates directly.
Run:  python plot_embedding_space.py   (writes embedding_space.png)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display; save to file
import matplotlib.pyplot as plt
from step1_data import chars, vocab_size

C = 2  # two dims so each token is literally an (x, y) point we can plot
rng = np.random.default_rng(7)
tok_emb = rng.normal(0, 0.02, size=(vocab_size, C))   # (26, 2)

print("With C=2, each row is an (x, y) coordinate:")
for i in range(5):
    print(f"  {chars[i]!r:5s} id {i:2d} -> (x={tok_emb[i,0]:+.4f}, y={tok_emb[i,1]:+.4f})")

# --- plot ---
plt.figure(figsize=(7, 7))
plt.scatter(tok_emb[:, 0], tok_emb[:, 1], s=10, color="lightgray")
for i, ch in enumerate(chars):
    label = {"\n": "\\n", " ": "spc"}.get(ch, ch)
    plt.annotate(label, (tok_emb[i, 0], tok_emb[i, 1]), fontsize=11)
plt.axhline(0, color="#ddd", lw=0.8); plt.axvline(0, color="#ddd", lw=0.8)
plt.title("Token embedding space at INIT (C=2)\nrandom now -> training moves the dots")
plt.xlabel("feature 0  (axis 0)"); plt.ylabel("feature 1  (axis 1)")
plt.tight_layout()
plt.savefig("embedding_space.png", dpi=110)
print("\nsaved embedding_space.png")

# --- geometry is what matters, and it works at ANY dimension ---
def dist(a, b): return float(np.linalg.norm(a - b))
from step1_data import stoi
print("\nDistances between points (this is the 'meaning' signal):")
print(f"  'a'<->'b': {dist(tok_emb[stoi['a']], tok_emb[stoi['b']]):.4f}")
print(f"  'a'<->'o': {dist(tok_emb[stoi['a']], tok_emb[stoi['o']]):.4f}")
print("(random now, so distances are meaningless -- after training, similar")
print(" tokens would sit CLOSE together; that closeness is what the model uses.)")
