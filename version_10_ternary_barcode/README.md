# version_10_ternary_barcode — bits that can abstain

v09's autopsy in one line: **a meter is not automatically a bit.** Binary
barcodes force every word to pick a side on every axis, so the neutral mass
coin-flips — that one flaw showed up in all three v09 failures (the good-bad
query undefined at sign-stability 0.875, fidelity diluted at large K, the
composition near-miss at 6/15). v10 makes one change and re-runs everything:

> bits live in **{−1, 0, +1}** with a per-axis **certified dead zone** —
> a word commits to a side only when its projection clears τᵢ, and
> abstains otherwise.

τᵢ is calibrated on split-half measurement data only (smallest τ on a fixed
z-grid where committed words' half-A/half-B signs agree ≥ 95%, axis kept only
if coverage ≥ 10%) — no downstream test signal touches the calibration. A new
word's code needs only its own vector plus the frozen (μ, σ, τ) table.
Ternary bundling gets abstention-on-conflict for free: +1 + −1 = 0 instead of
a coin flip. Sparse ternary is also the native regime of VSA memories.

Substrate: v09's extraction, untouched (`../version_09_semantic_barcode/
vocab_vectors.npz`, 1027 words, layer chosen blind). Probes: v09's 15,
verbatim, same bar — the rescue shot is earned only on identical terms.

| file | what it is |
|---|---|
| `ternary.py` | dead-zone calibration, ternary/binary code builders (paired on identical axes), sparse-aware cosine. |
| `step1_bank.py` | T0: certify the bank; both v08 meters must make it back in. |
| `step2_fidelity.py` | T1: ternary vs binary at equal K (paired), SimHash reference. |
| `step3_compose.py` | T2: the 15 v09 probes with ternary bundling. |
| `step4_lattice.py` | T3: committed-bits-only bipolar records (D=8192), threshold-unbind read rule θ_w = ½√(2/πn_w), exact three-way recovery + the rescued meter query. |
| `falsify.py` | T0–T3 verdict. |

## Pre-registered gates

- **T0**: ≥ 256 certified axes AND good-bad AND someone-something certified.
- **T1**: ternary ρ > binary ρ at K=256 (paired, same axes).
- **T2**: ≥ 7/15 probes hit top-20, p < 0.05 vs 200-run null.
- **T3**: exact ternary bit recovery ≥ 0.95 AND query precision & recall ≥ 0.95.

Results: see [`FINDINGS.md`](FINDINGS.md).
