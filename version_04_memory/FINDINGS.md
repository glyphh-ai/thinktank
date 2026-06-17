# version_04_memory — findings

Idea: encode each glyph's meaning as a VSA hypervector (JSON-as-letters:
Role/Segment/Layer/Cortex), project it to the embedding dim, and seed the GPT's
token embeddings with it ("use the glyph, not the generic char").

## Encoder (vsa.py) — works as specified
- Role=bind(key,value); each level = bundle of bind(name, child); paths are
  addressable by unbinding the key chain (a-sound recovered cos .52 vs .04).
- Similarity reflects structure: b/d .80, p/q .82, vowels cluster, unrelated ~.28.

## Using it in the loop (train_meaning.py, toy corpus, 3 seeds, 600 steps)
```
init             val loss(mean±std)   val acc   ->90% @
random              0.209 ± 0.006      91.9%       200
VSA-meaning         0.201 ± 0.007      92.3%       208
```
Within-noise tie: structured init neither helps nor hurts here.

## The real result: training erases the imposed structure
```
            cos(b,d)  cos(p,q)  cos(a,e)  cos(a,k)
at init       +0.74    +0.83    +0.50    +0.19
post-train    +0.13    +0.13    +0.06    +0.07
```
The loss reshapes embeddings to minimize next-char error, decorrelating them and
overwriting the meaning prior. With enough data signal, the model learns its own
geometry and discards the structured one -> "meaning as weighted vector" ends up
without the meaning.

## Frozen meaning channel (compare_frozen.py): trainable + fixed meaning, same slots
representation[id] = tok_emb[id] (trainable) + frozen_meaning[id] (never updates)
```
config                         val loss(mean±std)   val acc   ->90% @
trainable only                    0.209 ± 0.006      91.9%       200
trainable + frozen meaning        0.211 ± 0.002      92.0%       200
```
Dead tie. Freezing keeps the meaning from washing out, but the model just
learns to ignore it -- on a saturated task it doesn't need the prior. (Lower
variance is the only effect: the fixed substrate makes runs more consistent.)

## Conclusion so far
On this small saturated corpus a structured meaning prior is inert, whether
seeded-then-trained (washes out) or frozen (ignored). A tie is expected: with
enough data signal the model learns its own representation. The prior can only
matter where the model CAN'T learn it from data.

## WHY read-out (why.py) — the actual goal: decode the pick as meaning
Not generation, not a prior on training. The model trains normally (weights
only); when it picks the next glyph we align that pick to the FIXED meaning
vector in the same vocab slot and unbind the role paths to read WHY:

```
after 'to b'    -> 'e'  why sound: front, vowel       audible(vowel-ness) +0.23
after 'and '    -> 'b'  why sound: plosive, voiced    audible +0.04
after 'questio' -> 'n'  why sound: nasal, voiced      (shape: arch -- which n is)
after 'or to '  -> 't'  why sound: plosive, unvoiced  audible +0.05
after 'them.'   -> '\n' why sound: silence            audible +0.01
```
Every decode is correct, and the audible score cleanly separates vowels (.23)
from consonants (.04) from silence (.01). The fixed meaning store is the memory;
aligning a decision to it explains the choice in human terms. This is the part
of the idea that WORKS -- because it's introspection, not a training/gen aid.

## What would make the structure matter (untested)
- FREEZE tok_emb (don't train it): forces the model to USE the meaning geometry;
  tests whether the structure is sufficient, not just whether it survives.
- Low-data / few-shot or held-out-glyph generalization: a structured prior helps
  most when data is too scarce to learn a representation from scratch.
- A task that actually rewards the structure (e.g., predicting glyph properties).
