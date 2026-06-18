# version_05_lattice — findings

Idea: the per-step embedding trajectory (a Markov chain in the full training
state; a non-Markovian projection per char) is the "history of learning". Encode
each char's drift-between-steps into a history vector and read it as the "why".

## trajectory.py (toy corpus, 600 steps)

### Magnitude story (strong): history encodes frequency/difficulty
```
corr(frequency, path_length)   = -0.739   # rare chars travel much farther
corr(frequency, straightness)  = -0.527   # frequent chars wander, end near start;
                                          # rare chars move straight & far
history-vs-final embedding sim corr = -0.120   # history ~orthogonal to final weight
```
=> the trajectory carries info the FINAL weight discards: a char's role in the
data (how often / in how many contexts it appears).

### Category story (weak): does history cluster chars by what they ARE?
```
representation        sep    within  between
directional history +0.029   0.015   -0.014   weak category signal, noisy NN
magnitude history   -0.007   0.921    0.927   none (just the shared LR schedule)
final embedding     +0.076   0.042   -0.035   BEST at category
```

## Verdict
The learning history is a real, DISTINCT signal -- but it's a frequency/difficulty
lens, not a semantic one. For "what kind of char", the final embedding wins
(+0.076 vs +0.029). The history's unique value: it encodes data-role information
(frequency, contextual diversity) that the final weight throws away, and it's
~orthogonal to the final embedding. So "the why from history" = "how common / how
hard this was to learn", not "what it means".

Connection: the embedding sequence IS a Markov chain in the full state (weights +
Adam moments + RNG); a single char's trajectory is a projection of it -> generally
non-Markovian, which is exactly why its history leaks hidden-state info the final
weight lacks (the augmented-state / Nekrasov point).

## Ada IN the loop (ada_loop.py) — the synthesis that works

Word-level GPT over a fact corpus, two heads sharing the trunk: predict the
next word (LM) AND the next word's Ada slot (one of the 7-layer x 33-role
lattice, vendored from glyphh-ai/adaL). Ada slot is a TARGET in the loss:
  total = LM_CE + lambda * ROLE_CE
Dual-head backprop gradient-checked (1.2e-7).

```
config                     LM next-word acc   Ada role acc
LM only (lambda=0)               64.0%            4.4%   (role head ~chance)
Ada in the loop (lambda=1)       63.9%           90.5%   (model LEARNED the slots)
```
In-loop "why" (model emits word + its Ada role per step): carol lives in ->
austin [spatial.location]; the parrot is -> green [perceptual.color]; jack is
-> fifty [temporal.age] — all correct.

Verdict: putting Ada's schema in the LOSS makes it load-bearing (role acc
4.4 -> 90.5) at zero LM cost, and the "why" is GENERATED in the loop, not read
out. This resolves v04: meaning as a frozen INPUT gets ignored (tie); meaning
as a TARGET in the gradient gets learned. The fix is the gradient, not the input.
(Caveat: small templated corpus -> role task is easy; this proves the MECHANISM
is learnable and free, not that it's hard. In production Ada's LLM enricher
supplies the slot labels at write time.)

## Proof it really learns the STRUCTURE (prove.py) — fair tagging + killer baseline

Genuine role ambiguity: the same word fills different Ada slots by context
(city = location after "lives in" / origin after "is from"; pet = object after
"owns a" / entity.name in "the <pet> is..."). 14 ambiguous words. Fair tagging
setup: model and a word->majority-role LOOKUP baseline both see the same token;
only the model has context.

```
role accuracy (fair tagging)   overall   AMBIGUOUS   unambiguous
word->role lookup (baseline)     91.8%      58.7%        100.0%
Ada-in-the-loop model            99.2%      96.1%        100.0%
```

On ambiguous tokens the model beats lookup by +37.4 (96.1 vs 58.7). Lookup is
pinned at the majority-class rate because the word alone can't decide the role;
the model can only win by USING CONTEXT -> it genuinely learned Ada's
context-dependent slot assignment, not a word->role table. (This is exactly what
Ada's enricher does at write time, now learned in-loop at no LM cost.)

Caveat: synthetic corpus with local cues; proves the mechanism is real and
context-driven, not that messy real text is solved.

## Big test: in-loop on Ada's REAL data at scale (big_loop.py)

glyphh-ai/adaL real enricher output, 3 domains (chemistry/philosophy/law):
14,000 facts -> 151,935 tokens, vocab 8,002, 5,392 ambiguous words. ~30x the toy.

```
role accuracy (fair tagging)   overall   AMBIGUOUS   unambiguous
word->role lookup (baseline)     69.8%      63.4%        99.8%
Ada-in-the-loop model            90.0%      88.1%        98.8%
```
ambiguous-token gap: model 88.1% vs lookup 63.4% (+24.7).
LM next-word acc: lambda=1 36.3% vs lambda=0 35.1% -> role objective still FREE.

Verdict: the toy result HELD on real cross-domain data at 30x scale. Margin
shrank (+37 toy -> +24.7 real) as expected (messier vocab, <unk>, 5k+ ambiguous
words) but did not collapse -- the model disambiguates real domain terms by
context, decisively beating memorization, at zero LM cost.

Still open (honest): (1) labels are clean (from Ada's structured fields), so
enricher label-NOISE is untested; (2) 60k-param numpy model, not frontier scale;
(3) we proved the structure is LEARNED, not yet that emitting it BUYS something
downstream (factuality / refusal / control) -- the next real test.
