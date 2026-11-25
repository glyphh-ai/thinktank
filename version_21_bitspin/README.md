# version_21_bitspin — a language model made of bits and counters

The idea (Chris, 2026-11-08): train on the corpus as raw bits, treat each
bit as a half-turn (0 or π), give position by rotation and attention by
angle (XOR + popcount), and store meaning in counters behind the bits, so
the model learns on every forward pass and a correction moves only the
fact it corrects.

> **Can a model whose weights are counters learn new facts from the forward
> pass alone, take corrections, and do it without the forgetting a regular
> transformer pays when it is updated online?**

## Prior art (so we don't re-chase it)

| work | what it does | what it leaves open |
|---|---|---|
| **Bitformer** (2023) | attention score = XOR/Hamming between binary queries and keys | still trained by backprop |
| **BiT** (NeurIPS 2022), **BiPFT** | fully binarized transformers at practical accuracy | weights fixed after training |
| **BitNet b1.58** (Microsoft, 2024) | ternary weights match full precision at 3B | backprop with a real-valued shadow weight |
| **Gray-PE** (spiking transformers, 2025), **BiHDTrans** | binary / hypervector position and attention | classification, not generation |
| **Titans**, **TTT layers** (2024–25) | a memory module that updates during the forward pass | only the bolted-on memory learns; the main weights don't |
| **Kanerva SDM** (1988), Bricken & Pehlevan (2021) | counters at binary addresses; attention ≈ SDM | not run as an LM that learns while reading |

Nobody had a bits-in, bits-out LM whose only learned state is counters that
the forward pass itself updates, tested against a transformer on learning
and correcting facts online.

## The model (`bitspin.py`)

| piece | what it is |
|---|---|
| byte code | 1024-bit vector made from the byte's own 8 bits (XOR of one random vector per set bit) |
| rotation | a byte k steps back is rolled k words: position = angle |
| exact heads | XOR-bind the rotated bytes at lags 1..n (n = 1,2,3,4,5,6,8) → address |
| soft (angle) heads | majority-bundle the rotated bytes, sample k bits → address; two contexts collide with probability (1 − angle/π)^k |
| word heads | XOR-bind the last 1, 2 or 3 words, each turned 7k bits for k words back |
| memory head | the same over the last 6 words; it sits outside the mix and refines the final guess by its own counts alone, `p = (n1 + p) / (n + 1)`, so a fact lives only in its own counters |
| counters | every address holds (n0, n1) for the next bit: the predicted bit is their sign, the size is the trust; learning = +1 on the bit that came; on the memory head a contradiction halves the old count |
| trust | read off the counter itself (never contradicted and seen m times, or mixed), not off a shared table |
| mix | exact heads form a back-off chain (Witten-Bell at bit level); the chain, soft and word heads are mixed by a small weight per (head, trust level) |

No real-valued weights apart from that mix (6 × 12 numbers), no backprop,
no gradient. Text goes in as bits and comes out one bit at a time.

## Steps

| step | what it measures |
|---|---|
| `step1_lm.py` | bits/char on held-out *Pride and Prejudice* after learning from five novels (4.0 MB), the same protocol as version_19 step2 (order-5 n-gram: 1.900 adapting) |
| `step2_transformer.py` | a byte-level transformer (MLX, 10.9M params, ctx 256, 8000 AdamW steps) on the same data, for the same number |
| `step3_adjust.py` | 300 invented facts ("Piheba Doti keeps a hawk named Nipira."): recall after 1–3 showings, then a third corrected 1–3 times; damage = bits/char on 50k chars of *Pride and Prejudice* after each phase. Bitspin learns by running forward; the transformer by 1 or 3 Adam steps per fact at lr 1e-4, 3e-4, 1e-3 |

Run with `.venv/bin/python` (a link to version_19's env: numpy + MLX).
Results: `FINDINGS.md`.
