# FINDINGS — version_19_glyph

## step2: English compresses into pixels at 12.6 chars/pixel, exactly

Prior: a numpy order-5 byte model (Witten-Bell blend), trained on 4.0 MB from
five Gutenberg novels and adapting as it reads. Test: 200k chars of *Pride and
Prejudice*, held out. Every canvas is written to PNG, read back from disk and
compared byte-for-byte.

| codec | bits/char | chars/pixel | exact |
|---|---|---|---|
| **glyph stream** (126×126 canvas) | **1.900** | **12.63** | yes |
| **glyph tiles** (841 paragraphs, random access) | 2.250 | 10.67 | yes |
| bz2 -9 | 2.316 | 10.36 | – |
| lzma -9e | 2.614 | 9.18 | – |
| zlib -9 | 2.960 | 8.11 | – |
| typeset font (8×16 mono) | – | 0.02 | – |

Order sweep (stream): order 4 → 1.999, **order 5 → 1.900**, order 6 → 1.916.

**Read:** the whole 200k-char excerpt fits in a 126×126 image. That's about
630× denser than the typeset pages DeepSeek-OCR and Glyph feed their
encoders. Tile mode keeps every paragraph independently readable at an 18%
cost. The canvas looks like noise, and that's correct: a perfect code has
maximum entropy. It's also the catch for G3: a vision model can't "read"
this canvas the way it reads a font. It has to share the codec's prior.

## step2b: a real LM as the prior reaches 20 chars/pixel, exactly

Same 199k chars of *Pride and Prejudice*, prior = Qwen3-0.6B (MLX 4-bit),
47,006 tokens, PNG round-trip, byte-exact.

| prior | bits/char | chars/pixel | exact | speed (M2 Max) |
|---|---|---|---|---|
| n-gram order-5, stream | 1.900 | 12.63 | yes | seconds |
| **Qwen3-0.6B, stream** (100×100 canvas) | **1.202** | **19.97** | yes | 255s encode, 257s decode |
| Qwen3-0.6B, tiles (random access) | 1.626 | 14.76 | yes | 683s / 701s |

**Unseen text** (the thinktank's own 2026 write-ups, 127k chars, which
Qwen can't have trained on). Markdown full of tables and numbers is harder
than a novel, but it still decodes byte-exact:

| prior | bits/char | chars/pixel | exact |
|---|---|---|---|
| n-gram order-5 (trained on novels), stream | 3.103 | 7.73 | yes |
| **Qwen3-0.6B, stream** (93×93 canvas) | **1.621** | **14.80** | yes |
| Qwen3-0.6B, tiles | 2.203 | 10.89 | yes |

Memorization of *Pride and Prejudice* is at most part of the 20 vs 14.8 gap;
the domain shift is the rest. **About 15 chars/pixel is the honest number for
new text.**

**What it buys, honestly:** exact storage about 6–7× smaller than raw text.
That's known LLM compression (ts_zip, "Language Modeling Is Compression")
with pixels as the container. The read is serial at about 780 chars/s, and
the canvas is noise to any model that doesn't share the prior. It doesn't
make context cheaper to *think with*. That's step 4's job.

## step4a: a parallel read is ~780k chars/s, but this first pass is undertrained

Writer/reader pair, 64-char chunks, one forward pass reads the whole patch,
true 8-bit pixels, held-out *Pride and Prejudice*, 4,000 steps each.

| P | chars/px | char acc | chunk exact | read |
|---|---|---|---|---|
| 8 | 1.0 | 0.980 | 0.410 | 769k chars/s |
| 4 | 4.0 | 0.860 | 0.000 | 771k chars/s |
| 3 | 7.1 | 0.615 | 0.000 | 773k chars/s |
| 2 | 16.0 | 0.168 | 0.000 | (learned nothing) |

**Read:** speed is the win: about 1,000× the serial decode, as predicted.
Density isn't settled yet. P=8 gives 24 bits per char, far more than
needed, and still only reaches 98%, so this is under-training, not a
capacity wall. A 4× longer run on P=8 and P=4 is queued. The shape so far
(clean at 1, degrading by 4, gone by 16) fits the ~4 chars/pixel ceiling.

**4× longer (16,000 steps):**

| P | chars/px | char acc | chunk exact | read |
|---|---|---|---|---|
| 8 | 1.0 | **0.9973** | **0.912** | 604k chars/s |
| 4 | 4.0 | 0.8880 | 0.0005 | 520k chars/s |

At 1 char/pixel the parallel glyph is close to clean: 91% of 64-char chunks
come back exact in one glance. At 4 chars/pixel, 4× the training moved
character accuracy only from 86% to 89%. That's a wall, not under-training.
6 bits per char is in principle plenty (English is about 2 bits/char with
context), so the limit is this architecture's single linear bottleneck, not
information. **The practical parallel ceiling today is between 1 and 4
chars/pixel, and my ~4 estimate looks optimistic.** Getting past it would
take a convolutional/patch-structured writer, not more steps.

## step4b: the attractor canvas finds the right memory almost every time

Canvas = one P=4 glyph patch per stored 64-char passage. Read = write a
partial cue and settle once against the whole canvas (modern Hopfield ≡
attention), then decode. Memories are held-out *Pride and Prejudice* chunks,
with novel text added as distractors past 11k. 4,000 steps.

| memories | canvas px | cue | basin hit | exact recall |
|---|---|---|---|---|
| 1,000 | 16k | prefix 25–75% | 0.936 | 0.000 |
| 1,000 | 16k | 15% typos | 0.997 | 0.000 |
| 10,000 | 160k | prefix 25–75% | 0.986 | 0.000 |
| 10,000 | 160k | 15% typos | 1.000 | 0.000 |
| 50,000 | 800k | prefix 25–75% | 0.957 | 0.000 |
| 50,000 | 800k | 15% typos | **1.000** | 0.000 |

**Read:** the attractor half works. Given only a quarter to three quarters of
a passage, or a passage with 15% of its characters scrambled, one parallel
settle over 50,000 stored memories (a 0.8-megapixel image, 3.2M chars) lands
in the right basin 96–100% of the time. Store, read and pattern-complete are
one operation on one image.

Exact recall is zero for a boring reason: the settled patch goes through the
same undertrained P=4 reader as step4a (86% char accuracy, never a whole
chunk exact). Sample:

    cue:     'he whole family{soon aq LongbouYn; and\naddressed herself particu'
    settled: 'he whole samily soon at soug, urn, and ad re  ed her thd pertic '

So the basin is right and the reader is blurry. That's fixed by the same
thing step4a needs: a reader that's actually converged.

**Rerun at P=8 (1 char/px), 12,000 steps:**

| memories | canvas px | cue | basin hit | exact recall |
|---|---|---|---|---|
| 1,000 | 64k | prefix 25–75% | 0.932 | 0.712 |
| 1,000 | 64k | 15% typos | 1.000 | 0.713 |
| 10,000 | 640k | prefix 25–75% | 0.991 | 0.842 |
| 10,000 | 640k | 15% typos | 1.000 | 0.857 |
| 50,000 | 3.2M | prefix 25–75% | 0.980 | 0.853 |
| 50,000 | 3.2M | 15% typos | **1.000** | **0.889** |

    cue:     'he whole family{soon aq LongbouYn; and\naddressed herself particu'
    settled: 'he whole family soon at Longbourn; and\naddressed herself particu'

**Read:** it works. From half a passage, or one with 15% of its characters
scrambled, one parallel settle over 50,000 memories (3.2 million chars in a
3.2-megapixel image) returns the whole passage **character-perfect 85–89%**
of the time and picks the right memory 98–100% of the time. Exact recall
now tracks the reader's own one-glance exactness (91% in step4a), so the
reader is the remaining limit, not the attractor. The 1k row is lower,
probably because the first 1,000 chunks are the book's opening and front
matter, which read worse (not yet checked). It isn't a capacity effect,
since recall *rises* with more memories.

## step5: the exploded-window test is a null. The task didn't train for either arm

Question model trained from scratch (4 blocks, d=192) to answer "Where does
X live?" (recall) and "How many live in Y?" (aggregate) over N invented
facts. Names are random per episode. Trained on N = 4–128 for 12,000 steps.

| arm | N=16 recall | N=16 aggregate | N=64 recall | N ≥ 256 |
|---|---|---|---|---|
| canvas (1 token per fact, from its 8×8 patch) | 0.000 | 0.000 | 0.000 | 0.000 |
| window (raw chars, 4,096-char window) | 0.373 | 0.010 | 0.010 | 0.000 |

**Read:** this says nothing about canvases yet. Even the plain text window
fails at N=64, where everything fits inside its window, so the from-scratch
question model never learned reliable lookup, let alone counting. The
canvas arm is strictly harder: its writer only gets signal through the
answer loss, and it learned nothing. The design was too ambitious for a
from-scratch model at this size.

**Redesign:** stop learning the memory through the answer. Reuse the
trained step4b writer, whose canvas already lands the right memory 98–100%
of the time, and do aggregation *in the settle itself*: cue "? lives in
Y.", and count how many memories the attention puts its mass on. That needs
4b retrained with masks anywhere in the passage, not only suffix masks.
