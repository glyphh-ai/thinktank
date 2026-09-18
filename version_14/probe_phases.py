"""
Probes for 3-phase π-frequency attention.
==========================================
Not about loss — about whether the mechanism works the way we think it does.

1. Phase diversity:  Do the 3 phases produce distinct attention patterns?
2. Per-position acc:  Is accuracy more uniform across the context window?
3. Weight transfer:   Do weights trained at N phases work at M phases?
4. Attention viz:     Side-by-side attention heatmaps per phase.

Run:  python probe_phases.py [params_file]   (default: strat0_params_3phase.npz)
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step1_data import val_data, vocab_size, BLOCK_SIZE, decode, stoi, itos

C = 128
BLOCK = BLOCK_SIZE
EPS = 1e-5


# ---- load params + positional encoding ------------------------------------

def multi_phase_pi_pos(T, C, n_phases):
    pos = np.arange(T)[:, None].astype(np.float64)
    dim = np.arange(C)[None, :].astype(np.float64)
    freq = 1.0 / (np.pi ** (2.0 * (dim // 2) / C))
    return np.stack([
        np.where(dim % 2 == 0,
                 np.sin(pos * freq + 2.0 * np.pi * k / n_phases),
                 np.cos(pos * freq + 2.0 * np.pi * k / n_phases))
        for k in range(n_phases)
    ])


def forward_with_attention(p, x_ids, n_phases):
    """Forward pass that returns per-phase attention matrices."""
    B, T = x_ids.shape
    tok = p["tok_emb"][x_ids]
    positions = multi_phase_pi_pos(T, C, n_phases)

    # layernorm
    mu = tok.mean(-1, keepdims=True)
    var = tok.var(-1, keepdims=True)
    ln = (tok - mu) / np.sqrt(var + EPS) * p["g1"] + p["b1"]

    phase_atts = []
    phase_outs = []
    mask = np.tril(np.ones((T, T), dtype=bool))

    for phi in range(n_phases):
        x = ln + positions[phi]
        q = x @ p["Wq"]
        k = x @ p["Wk"]
        v = x @ p["Wv"]
        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(C)
        scores = np.where(mask, scores, -1e9)
        scores -= scores.max(-1, keepdims=True)
        e = np.exp(scores)
        att = e / e.sum(-1, keepdims=True)
        out = np.matmul(att, v)
        phase_atts.append(att)
        phase_outs.append(out)

    combined = sum(phase_outs) / n_phases
    h1 = tok + combined @ p["Wo"]

    # ln2 + mlp
    mu2 = h1.mean(-1, keepdims=True)
    var2 = h1.var(-1, keepdims=True)
    ln2 = (h1 - mu2) / np.sqrt(var2 + EPS) * p["g2"] + p["b2"]
    hid = ln2 @ p["W1"] + p["b1m"]
    act = hid * (hid > 0)
    h2 = h1 + act @ p["W2"] + p["b2m"]

    # lnf + head
    muf = h2.mean(-1, keepdims=True)
    varf = h2.var(-1, keepdims=True)
    lnf = (h2 - muf) / np.sqrt(varf + EPS) * p["gf"] + p["bf"]
    logits = lnf @ p["Wh"] + p["bh"]

    return logits, phase_atts


def get_batches(n_batches, B=64, T=BLOCK, seed=999):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(n_batches):
        ix = rng.integers(0, len(val_data) - T, size=B)
        x = np.stack([val_data[i:i + T] for i in ix])
        y = np.stack([val_data[i + 1:i + 1 + T] for i in ix])
        batches.append((x, y))
    return batches


# ============================================================================
# PROBE 1: Phase diversity — how different are the attention patterns?
# ============================================================================
def probe_phase_diversity(p, n_phases=3, n_batches=10):
    print("=" * 70)
    print(f"PROBE 1: Phase diversity ({n_phases} phases)")
    print("=" * 70)

    batches = get_batches(n_batches)
    pairwise_diffs = []

    for x, y in batches:
        _, atts = forward_with_attention(p, x, n_phases)
        for i in range(n_phases):
            for j in range(i + 1, n_phases):
                diff = np.abs(atts[i] - atts[j]).mean()
                pairwise_diffs.append(diff)

    mean_diff = np.mean(pairwise_diffs)
    print(f"  mean |att_i - att_j| across all phase pairs: {mean_diff:.4f}")
    print(f"  (0 = identical patterns, higher = more diverse)")

    # also measure: what fraction of positions have a different argmax across phases?
    argmax_disagree = []
    for x, y in batches:
        _, atts = forward_with_attention(p, x, n_phases)
        argmaxes = np.stack([a.argmax(-1) for a in atts])  # (n_phases, B, T)
        for i in range(n_phases):
            for j in range(i + 1, n_phases):
                disagree = (argmaxes[i] != argmaxes[j]).mean()
                argmax_disagree.append(disagree)

    print(f"  fraction of positions where phases disagree on argmax: {np.mean(argmax_disagree):.1%}")
    return mean_diff


# ============================================================================
# PROBE 2: Per-position accuracy — is it more uniform?
# ============================================================================
def probe_per_position_accuracy(p, phase_configs, n_batches=20):
    print("\n" + "=" * 70)
    print("PROBE 2: Per-position accuracy")
    print("=" * 70)

    batches = get_batches(n_batches)
    T = BLOCK

    results = {}
    for n_phases in phase_configs:
        correct = np.zeros(T)
        total = np.zeros(T)
        for x, y in batches:
            logits, _ = forward_with_attention(p, x, n_phases)
            preds = logits.argmax(-1)
            for t in range(T):
                correct[t] += (preds[:, t] == y[:, t]).sum()
                total[t] += x.shape[0]
        acc = correct / total
        results[n_phases] = acc
        uniformity = acc.std() / acc.mean()
        print(f"  {n_phases:2d}-phase: mean acc {acc.mean():.1%}, "
              f"std {acc.std():.1%}, "
              f"CV {uniformity:.3f} (lower = more uniform)")

    # plot
    plt.figure(figsize=(10, 5))
    for n_phases, acc in results.items():
        plt.plot(acc, label=f"{n_phases}-phase", alpha=0.8)
    plt.xlabel("Position in context window")
    plt.ylabel("Next-word accuracy")
    plt.title("Per-position accuracy by phase count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("probe_per_position.png", dpi=120)
    print("  saved -> probe_per_position.png")
    return results


# ============================================================================
# PROBE 3: Weight transfer — do weights work at different phase counts?
# ============================================================================
def probe_weight_transfer(p, phase_configs, n_batches=20):
    print("\n" + "=" * 70)
    print("PROBE 3: Weight transfer (trained at 3 phases, tested at N)")
    print("=" * 70)

    batches = get_batches(n_batches)

    for n_phases in phase_configs:
        losses = []
        accs = []
        for x, y in batches:
            logits, _ = forward_with_attention(p, x, n_phases)
            # loss
            flat = logits.reshape(-1, vocab_size)
            flat = flat - flat.max(1, keepdims=True)
            probs = np.exp(flat)
            probs /= probs.sum(1, keepdims=True)
            yf = y.reshape(-1)
            loss = -np.log(probs[np.arange(yf.shape[0]), yf] + 1e-12).mean()
            acc = (logits.argmax(-1) == y).mean()
            losses.append(loss)
            accs.append(acc)
        print(f"  {n_phases:3d}-phase: val loss {np.mean(losses):.3f}  "
              f"val acc {np.mean(accs):.1%}")


# ============================================================================
# PROBE 4: Attention heatmaps per phase
# ============================================================================
def probe_attention_viz(p, n_phases=3):
    print("\n" + "=" * 70)
    print(f"PROBE 4: Attention heatmaps ({n_phases} phases)")
    print("=" * 70)

    # grab a single story-like sequence from val
    rng = np.random.default_rng(42)
    start = rng.integers(0, len(val_data) - BLOCK)
    x = val_data[start:start + BLOCK].reshape(1, BLOCK)
    words_shown = [itos[int(x[0, t])] for t in range(BLOCK)]

    _, atts = forward_with_attention(p, x, n_phases)

    n_show = min(n_phases, 6)
    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5))
    if n_show == 1:
        axes = [axes]

    T_show = min(16, BLOCK)
    for i in range(n_show):
        ax = axes[i]
        att = atts[i][0, :T_show, :T_show]
        im = ax.imshow(att, cmap="Blues", vmin=0, vmax=att.max())
        phase_deg = int(360 * i / n_phases)
        ax.set_title(f"Phase {i} ({phase_deg}°)")
        ax.set_xticks(range(T_show))
        ax.set_xticklabels(words_shown[:T_show], rotation=90, fontsize=6)
        ax.set_yticks(range(T_show))
        ax.set_yticklabels(words_shown[:T_show], fontsize=6)

    plt.suptitle("Attention patterns per phase (same weights, different π-pos)")
    plt.tight_layout()
    plt.savefig("probe_attention_phases.png", dpi=120)
    print(f"  saved -> probe_attention_phases.png")
    print(f"  sequence: {' '.join(words_shown[:T_show])}")


# ============================================================================
# Run all probes
# ============================================================================
if __name__ == "__main__":
    params_file = sys.argv[1] if len(sys.argv) > 1 else "strat0_params_3phase.npz"
    print(f"Loading params from {params_file}\n")
    loaded = np.load(params_file)
    p = {k: loaded[k] for k in loaded.files}

    probe_phase_diversity(p, n_phases=3)
    probe_per_position_accuracy(p, phase_configs=[1, 3, 6, 12])
    probe_weight_transfer(p, phase_configs=[1, 2, 3, 6, 12, 36])
    probe_attention_viz(p, n_phases=3)

    print("\n" + "=" * 70)
    print("All probes complete.")
    print("=" * 70)
