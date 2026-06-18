"""
trajectory.py — the learning history of each char's embedding as the "why".
==========================================================================
The sequence of embedding states across training is a Markov chain (in the full
state: weights + Adam moments + RNG). A single char's embedding trajectory is a
PROJECTION of that chain -> generally non-Markovian, so its history carries
information the final weight discards.

We log every char's embedding at every step, take the TRANSITION INCREMENTS
(the drift between steps), denoise them into a per-char directional "history
vector" (the corpus vector of its weights), and test whether that history
clusters chars by what they ARE -- i.e. whether the history is a real "why".

Run:  python trajectory.py
"""

import numpy as np
import step3_gpt as G
import step1_data as D

STEPS = 600
PHASES = 6          # split training into phases; mean drift-direction per phase


def train_logging(steps=STEPS, seed=0):
    rng = np.random.default_rng(seed)
    p = G.init_params(rng)
    opt = G.Adam(p, lr=3e-3)
    brng = np.random.default_rng(0)
    traj = np.zeros((steps + 1, G.vocab_size, G.C))
    traj[0] = p["tok_emb"]
    for s in range(1, steps + 1):
        x, y = G.get_batch("train", brng, B=32)
        _, _, cache = G.forward(p, x, y)
        opt.step(p, G.backward(p, cache))
        traj[s] = p["tok_emb"]
    return traj


def history_vectors(traj):
    """Per-char directional history vector: increments -> unit directions ->
    mean direction per training phase -> flattened. Removes the shared LR-decay
    magnitude so what's left is HOW (which way) each char moved over time."""
    inc = np.diff(traj, axis=0)                              # (STEPS, V, C) transitions
    unit = inc / (np.linalg.norm(inc, axis=2, keepdims=True) + 1e-9)
    chunks = np.array_split(unit, PHASES, axis=0)            # phases over time
    phase_dir = np.stack([c.mean(0) for c in chunks], axis=1)  # (V, PHASES, C)
    return phase_dir.reshape(G.vocab_size, -1)               # (V, PHASES*C)


def cos_mat(M):
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return Mn @ Mn.T


def category(ch):
    if ch in "aeiou": return "vowel"
    if ch in " \n":    return "space"
    if ch in ",.:":    return "punct"
    return "cons"


def group_separation(S, cats):
    """mean within-category similarity minus mean between-category similarity.
    Higher => the representation separates the categories."""
    V = len(cats)
    win, bet = [], []
    for i in range(V):
        for j in range(V):
            if i == j: continue
            (win if cats[i] == cats[j] else bet).append(S[i, j])
    return np.mean(win) - np.mean(bet), np.mean(win), np.mean(bet)


def main():
    traj = train_logging()
    chars = [D.itos[i] for i in range(G.vocab_size)]
    cats = [category(c) for c in chars]

    hist = history_vectors(traj)          # directional history (the corpus vector)
    final = traj[-1]                      # final learned embedding
    mag = np.linalg.norm(np.diff(traj, axis=0), axis=2).T   # magnitude-only signature

    S_hist, S_final, S_mag = cos_mat(hist), cos_mat(final), cos_mat(mag)

    print(f"vocab={G.vocab_size}  C={G.C}  steps={STEPS}  phases={PHASES}")
    print("category counts:", {c: cats.count(c) for c in set(cats)}, "\n")

    print("Does the representation cluster chars by CATEGORY? "
          "(within - between similarity; higher = better)\n")
    print(f"{'representation':24} {'sep':>7} {'within':>8} {'between':>8}")
    for name, S in (("directional history", S_hist),
                    ("magnitude history", S_mag),
                    ("final embedding", S_final)):
        sep, w, b = group_separation(S, cats)
        print(f"{name:24} {sep:+7.3f} {w:8.3f} {b:8.3f}")

    # nearest neighbour in directional-history space for a few chars
    print("\nnearest chars by directional learning history:")
    for ch in ["a", "e", "t", " ", ".", "k"]:
        i = D.stoi[ch]
        order = np.argsort(-S_hist[i])
        nn = [chars[j] for j in order if j != i][:4]
        print(f"  {ch!r:>4} ({category(ch):5}) -> {[ (n, category(n)) for n in nn ]}")

    print(f"\nhistory vector shape (corpus vector per char): {hist.shape}")


if __name__ == "__main__":
    main()
