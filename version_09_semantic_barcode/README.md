# version_09_semantic_barcode — named bits: a barcode where every bit means something

Born from the v08 debrief: a certified prime-pair meter read at sign-only
precision is one **named bit**. K meters → a K-bit code in {−1,+1}ᴷ — which
is *natively* a bipolar hypervector: bundle = majority vote, bind = multiply,
similarity = Hamming. No encoding step between "semantic measurement" and
"lattice storage."

The math underneath is random-hyperplane LSH (SimHash): sign-of-projection
codes converge to cosine geometry as K grows, and *random* hyperplanes are
provably near-optimal for that. v08's H3 showed prime axes are thin,
specialized blades — so named bits should LOSE to SimHash at small K. The
question v09 asks is the one that matters for shipping:

> **How big is the fidelity tax for making every bit interpretable, at
> realistic width — and does the code stay compositional and
> lattice-storable?**

## The experiment

Substrate: same frozen `Qwen/Qwen2.5-0.5B`. New eval set: the top-1000
frequency words of English (google-10000 list) + probe words = **1027 words**
(committed as `wordlist.txt`), disjoint from the 61 primes that build the
axes.

| file | what it is |
|---|---|
| `vocab.py` | eval vocabulary + **15 pre-registered composition probes** (a + b → expected words), fixed before any barcode existed. |
| `step1_vocab_vectors.py` | v07's exact measurement protocol over the 1027 words (~12.3k forward passes, CPU) → `vocab_vectors.npz` (gitignored). |
| `barcode.py` | shared core: blind layer choice (split-half reliability on the eval set), axis construction, `sign(V @ axes.T)` barcodes. |
| `step2_mine.py` | the named axis bank: all 1830 prime-pair axes, certified by split-half **sign stability ≥ 0.90** on the eval set, ordered by greedy geometric decorrelation (no eval data in the ordering — no adaptive edge over SimHash). |
| `step3_fidelity.py` | **the fidelity-tax curve**: named bits vs SimHash vs random word-pair axes at K = 16…1024; Spearman vs full cosine geometry over all ~527k word pairs + top-10 neighbor Jaccard. |
| `step4_compose.py` | bundle(code(a), code(b)) → rank expected word among 1025; 15 probes vs 200-run random-input null. |
| `step5_lattice.py` | 256 named bits bound into ONE D=8192 record per word (`Σ bitᵢ·axis_idᵢ`); read back by unbinding; feature query "valence− AND animate+" answered by unbind only. |
| `falsify.py` | K1–K4 verdict. |

## Pre-registered kill criteria

- **K1 width**: ≥ 256 axes stable at ≥ 0.90 sign agreement across template
  halves — else BARCODE DEAD.
- **K2 fidelity**: at K=256, named ρ ≥ 0.80 × SimHash ρ (tax ≤ 20%) — else
  TAX TOO HIGH.
- **K3 composition**: ≥ 7/15 probes hit top-20, p < 0.05 vs null — else
  FAITHFUL BUT NOT COMPOSITIONAL.
- **K4 lattice**: bit recovery ≥ 0.99; unbind-only query precision and
  recall ≥ 0.95 — else CARRIER BROKEN.

## Why this isn't SimHash

SimHash bits are anonymous; only aggregate distance means anything. Here bit
*i* is a certified prime contrast (good−bad, someone−something, ...) — so
similarity is auditable ("close because they agree on valence, animacy, and
217 other named features"), queries are bitmasks instead of classifiers, and
bundled gists can be read feature-by-feature. The tax curve prices exactly
what that buys.

## Run

```
python vocab.py            # validate committed wordlist + probes
python step1_vocab_vectors.py   # ~1 h CPU -> vocab_vectors.npz
python falsify.py          # K1..K4 + verdict
```

Results: see [`FINDINGS.md`](FINDINGS.md).
