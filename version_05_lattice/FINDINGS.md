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
