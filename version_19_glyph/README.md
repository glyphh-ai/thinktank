# version_19_glyph — context as pixels: a model that reads *and writes* a canvas

The idea (Chris, 2026-10-31): store context history in the pixels of a large,
dense image, and train a model to use that image, not text tokens, as the
thing it generates from. A glyph in the true sense of the word.

> **Can a model keep its context as a dense pixel canvas, recall from it
> exactly, and generate by writing new pixels onto it, more cheaply per fact
> than text tokens?**

## Prior art (what's been tried, so we don't re-chase it)

| work | what it does | what it leaves open |
|---|---|---|
| **PIXEL** (Rust et al., ICLR 2023), "Language Modelling with Pixels" | Masked-patch LM trained only on *rendered text images*; no vocabulary at all. Robust to unseen scripts. | Encoder only. It reads pixels but doesn't generate. |
| **CLIPPO** (Tschannen et al., 2023) | One ViT for images and rendered text, no tokenizer. | Same: understanding, not generation. |
| **Pix2Struct** (Lee et al., 2023) | Pretrains on screenshots → HTML. | Output is still text tokens. |
| **PIXAR** (Tai et al., 2024) | First *autoregressive* pixel LM: generates rendered text as image patches. | Generation quality lagged text LMs badly and needed adversarial loss to get legible output. The canvas is still *rendered text*. |
| **DeepSeek-OCR** (Oct 2025), "Contexts Optical Compression" | Renders long context into an image and decodes it from vision tokens: ~97% recall at <10× compression, ~60% at 20×. Pitched explicitly as context memory with "fading" resolution for old turns. | Reads only. Answers in text. Lossy past 10×. |
| **Glyph** (Zhipu/Tsinghua, Oct 2025), "Scaling Context Windows via Visual-Text Compression" | Renders long text to pages and feeds a VLM: 3–4× fewer tokens, faster prefill, 1M-token tasks from a 128k model. | Same name as us. Reads only, and the canvas is still typeset text. |

**The common pattern:** everyone so far renders *human text* as an image and
gets compression from the vision encoder's dense continuous tokens. Nobody has
the model **write its own canvas in a learned, non-human pixel code** and
generate by extending that canvas.

## Why the gap matters (and the honest catch)

Rendering text doesn't add information. A typeset page is a *worse* encoding
of the same bytes. The 3–10× win comes entirely from replacing discrete
tokens (~17 bits each) with continuous vision tokens (thousands of floats
each). So the real question isn't "pixels vs text". It's this:

1. **Learned canvas vs rendered canvas.** If the model writes its own glyphs,
   it can pack far more than a font does: 2D locality, color channels as
   separate fields, and resolution as a natural importance/recency axis
   (DeepSeek's "fading memory").
2. **Exactness.** Every result above is lossy. Ada's thesis (v05, `frontier/`)
   is that the exact store wins. A glyph canvas is only interesting if recall
   stays exact at a compression text can't match, or if it degrades
   *gracefully and knowingly*.
3. **This is the barcode line in 2D.** v09/v10 made named-bit codes that are
   natively hypervectors. A glyph canvas is the same object laid out spatially:
   a patch is a code, the canvas is the lattice. That makes v19 a direct
   continuation, not a side quest.

## The experiment (numpy-first, like the rest of the line)

Substrate: the adaL enriched facts (`frontier/prepare_data.py` output) as
the context history.

| step | what it does |
|---|---|
| `step1_render.py` | **Baseline A, rendered canvas:** typeset facts into a bitmap (fixed mono font), cut into P×P patches. |
| `step2_codec.py` + `glyph.py` | **Arm B, the glyph language:** a predictive model of English plus arithmetic coding writes text into RGB pixels at 24 bits each, and the same model reads it back exactly. Stream mode uses one canvas; tile mode makes each paragraph self-delimiting for random access. **Done, see FINDINGS.md.** |
| `step3_recall.py` | Query → which patch → decode. Measures exact-fact recall vs **bits per fact** for A, B, and a plain text-token baseline. |
| `step4_write.py` | **Generation:** given the canvas, predict the *next patch* (the next fact or turn) in pixel space, then decode it. The part nobody has done on a learned canvas. |
| `step4a_parallel.py` + `glyph_net.py` | **Parallel glyph:** a learned writer/reader pair that reads a whole patch in one forward pass, sweeping density from 1 to 16 chars/px. |
| `step4b_attractor.py` | **Attractor canvas:** stored patches form one image, and a partial or noisy cue settles once against all of them (modern Hopfield ≡ attention). Store, read and pattern-complete are one operation. |
| `step5_window.py` | **Exploded window:** a question model cross-attends to every patch while it answers. It's tested on recall *and* aggregation ("how many live in X?"), trained on ≤128 facts and run at up to 4,096 (262k chars), against a 4,096-char text window. |
| `step5_fade.py` | Downsample old regions of the canvas and track how recall decays per fact age. |

## Pre-registered gates (fixed before any canvas exists)

| gate | pass condition |
|---|---|
| G1 capacity | Learned canvas B stores facts at **≥4× fewer bits/fact** than text tokens with **≥99% exact recall**. |
| G2 beats font | B beats rendered canvas A on bits/fact at equal recall by **≥2×**. Otherwise the "learned glyph" adds nothing over DeepSeek/Glyph. |
| G3 generation | Next-patch prediction decodes to a valid, schema-correct fact **≥80%** of the time on held-out continuations. |
| G4 graceful fade | Under 2× and 4× downsampling, recall loss is **monotone in age** and the model's confidence tracks it (it knows what it lost). |

**Kill criteria:** if G1 fails, the canvas is just a worse store than Ada, so
stop. If G1 passes but G2 fails, the prior art already owns it, so write it up
and stop. If G3 fails, it's a memory format, not a generation substrate, which
is still useful as Ada's cold tier.
