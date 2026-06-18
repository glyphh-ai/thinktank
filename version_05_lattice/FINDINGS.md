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
