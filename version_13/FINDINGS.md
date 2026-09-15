# FINDINGS — version_13

The constructive inversion of v07–v12: don't look for the primes inside a
trained model — AUTHOR the semantic space first, then make a model live in
it. Every word of a real corpus is judged by an LLM into a fixed, weighted,
NSM-prime lattice; the judged structure is bound into hypervectors; the
hypervectors replace the learned embedding layer of a from-scratch GPT.
All designs committed before results (`git log`).

## Strat 0 — the baseline must speak: PASS

Word-level from-scratch GPT (1 block, 1 head, C=128, T=32, 1.58M params,
hand-written backprop) on TinyStories-valid: 4.45M word tokens, vocab
5,375 @ freq>=10 (99.6% coverage). 8k Adam steps: **val loss 3.12, val acc
35%**, samples are grammatical story English with 1-block-shaped failures
(identity drift, plot wobble). Substrate earned; 3.12/35% is the ceiling
every later condition is measured against. version_0_base updated to this
same word-level baseline.

## Strat 1 — the lattice + the judge: BUILT, BACKWARDS PASSES

Schema (`strat1_schema.py`): 5 weighted layers > 6 segments > 13 roles;
every role's values drawn from the 65 NSM primes + "none". Fixed roles for
every word — codes comparable bit-for-bit.

VSA (`strat1_vsa.py`): D=100k bipolar, role(x)value bind, weighted bundle
to one ternary cortex vector per word; all atoms deterministic (sha256
seeds — the geometry rebuilds from the schema alone). Two structural
erasure bugs found by self-test and fixed: (1) sign() at intermediate
levels deletes any role lighter than a sibling — carry floats to the top;
(2) exact weights allow pointwise dominance (|1.0±0.9±0.8| >= 0.7 > 0.63)
— per-role coordinate jitter turns weights into probabilities of
controlling a coordinate. After both: **13/13 roles decode backwards from
a single cortex vector**, decode score tracks judged weight (weights
survive the round trip).

Judge (`strat1_judge.py`): claude-haiku-4-5, schema-enforced structured
output, all **5,374 words judged** (readings.jsonl, vocab-sha-locked to
the corpus; `lattice_bank.npz` rebuildable). Readings are semantically
right (huge -> big+very; die -> die, happen:0.7, bad:0.5). Judge quirks
observed: animals judged "someone"; motion words repeatedly reached for
spatial=move (fixed by prompt rule). Known ceiling: 13 roles => only
3,877 distinct codes; largest identical cluster 243 words.

## Strat 2 — frozen lattice vs frozen random vs learned: FAIL (both gates), earned

Same model/data/seeds, only the embedding table varies. Lattice = cortex
vectors JL-projected 100k->128 (cosine preservation r=0.947), row-normed.

| table | val loss | val acc |
|---|---|---|
| learned | **3.235** | 34.4% |
| random frozen | 3.663 | 30.2% |
| lattice frozen | 3.861 | 26.2% |

G1 (lattice beats random by >=0.10): **FAIL, −0.197** — lost to noise.
G2 (within 0.15 of learned): **FAIL, +0.626**.

Diagnosis, measured: **identity collapse**. 35% of words share their exact
code with another word (243-word worst cluster); mean pairwise cosine 0.12;
effective rank 77/128. Random wins because random is a perfect IDENTITY
code; the lattice is a CATEGORY code. If K words share a code the input
forfeits log2(K) bits. v12's lesson recurring: an attribute/similarity
representation alone is structurally insufficient — here as LM input.

## Strat 3 — hybrid identity + semantics: PARTIAL (G1' PASS, G2' FAIL)

E = row_norm(a·lattice + b·id), id = deterministic unique unit vector per
word (a name tag with zero semantics by construction).

| table | val loss |
|---|---|
| learned | **3.235** |
| hybrid 1.0·lat + 0.5·id | **3.526** |
| hybrid 0.5·lat + 1.0·id | 3.575 |
| random (id only) | 3.663 |
| lattice only | 3.861 |

G1' (best hybrid beats random by >=0.05): **PASS, +0.138** — with identity
restored, authored semantics adds real signal, ~3x the gate. The
semantics-forward mix won: once every word is distinguishable, more
lattice is better. **The core claim is alive.**
G2' (within 0.15 of learned): FAIL, +0.291. Suspected remaining channels:
corpus co-occurrence (relatedness — v12's ghost, third appearance),
frequency-in-norm (erased by row normalization), 13-role resolution.

## Strat 4 — learned per-word gain: PARTIAL (G1'' PASS, G2'' FAIL)

E = s_w · row_norm(1.0·lat + 0.5·id); table frozen, only s (one scalar per
word, init 1.0, 5,375 params — each readable as a loudness) learns. Gain
gradient verified against finite differences (2.1e-07).

**val loss 3.473, acc 31.8%.** G1'' (improve strat 3 best by >=0.05):
**PASS, +0.052** — just clears. G2'' (within 0.15 of learned): FAIL,
+0.238. The gain vector is itself the predicted finding: loudest words are
the story scaffold (<eot> 2.99, moral 2.93, mr, saw, she, named, is), and
corr(gain, log frequency) = **0.41** — the frequency-in-norm channel is
real, one scalar per word buys ~18% of the remaining gap, and what's left
(0.24) is the co-occurrence/relatedness + 13-role-resolution channel that
scalars cannot carry.

## Where this leaves the program

The ladder, one variable at a time: lattice 3.861 -> random 3.663 ->
+identity 3.526 -> +frequency 3.473 -> learned 3.235. An authored, named,
NSM-grounded geometry is (a) buildable at corpus scale by an LLM judge for
~$10, (b) decodable BACKWARDS from single ternary vectors — every role
readable, weights included, (c) genuinely useful to a transformer as input
— but only ON TOP of word identity (+0.138) and frequency (+0.052), and
(d) not yet a full replacement for learned embeddings; the measured
residue (0.24) is the co-occurrence/relatedness channel plus 13-role
resolution — decomposed, named, and priced in val loss.
The program's recurring invariant, now seen from a third angle: attribute
structure IS similarity, and similarity alone is neither retrieval (v11),
relatedness (v12), nor an LM input (v13 strat 2). It composes; it does not
substitute.

## Reproduce

```
python strat0_train.py                  # baseline (~15 min CPU)
python strat1_judge.py                  # needs ANTHROPIC_API_KEY (~$10)
python strat1_encode.py                 # readings -> lattice_bank.npz
python strat2_train.py                  # 3-way gate (~45 min)
python -u strat3_train.py               # hybrid gate (~30 min)
python -u strat4_train.py               # gain gate (~15 min)
```
