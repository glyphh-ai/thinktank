# FINDINGS — version_20_precedent

## Verdict: the move objective did not earn its place

The bar written before any run was `move` beating `decision` by at least 10
points on unseen domains. It beat it by 1.6, inside the noise, and lost by 7.3
on the harder set. By the rule in the README, the idea dies here.

Qwen3-1.7B-4bit, LoRA, 1000 steps, 6000 examples, one seed, 600 cases per test
set. Decision accuracy; chance is 50%.

| | unseen | apply | distinguish | overrule | reverse | harder (5 attrs, 14 precedents) |
|---|---|---|---|---|---|---|
| nearest precedent | 29% | 100% | 5% | 11% | 0% | 29% |
| nearest + outcome | 50% | 100% | 0% | 0% | 100% | 50% |
| latest + outcome | 68% | 85% | 38% | 71% | 79% | 65% |
| **decision** | 81.2% | 88.0% | 68.0% | 82.7% | 86.0% | **78.8%** |
| **move** | 82.8% | 81.3% | 79.3% | 77.3% | 93.3% | 71.5% |
| **after** | **83.8%** | 94.0% | 72.7% | 77.3% | 91.3% | 77.0% |
| Sonnet, reading the page (48 cases) | 96% | 12/12 | 12/12 | 10/12 | 12/12 | not run |

Exact move named: `move` 64.3% unseen, 33.5% harder; `after` 57.2%, 31.2%.

## Against the predictions

1. Nearest precedent fails the hard moves: **held**.
2. `move` beats `decision` by 10 points: **failed**, +1.6 and −7.3.
3. `move` beats `after`: **failed**, −1.0 and −5.5. Naming the move first did
   not help the decision. On the harder set it hurt, likely because a wrong
   move comes first and the decision follows it.
4. A frontier model is hard to beat: **held**, 96% against 84%. Sonnet was told
   the world's rules in its brief and the trained arms had to infer them, and
   48 cases is a small sample, so read this as direction, not a measured gap.

## What is real

- A 1.7B model learned to reason from precedents on words and policies it never
  saw, 13 to 16 points above the best free rule (68%), and it holds at 72 to 79%
  on bigger pages. That is the plain `decision` arm, so it needs no new objective.
- `move` gave 79% on distinguish against 68% for `decision`, but that gap is
  gone on the harder set (70.0% against 68.7%), and it is one seed on 150 cases.
- `move` does produce a citation, but it is right only 64% of the time, and it
  often reaches the right decision with the wrong move (`overrule` where the
  gold is `distinguish`). The explanation is not yet trustworthy.

## What was not tested

- One seed. A 10-point gap is unlikely to be seed noise, but small differences
  above are not resolved.
- `move` validation loss was still falling at step 1000 (0.108 to 0.081), so a
  longer run might narrow the harder-set gap. `decision` had flattened.
- The write-back stream test (overrule changes retrieval, adaptation with no
  weight update) was never run, because the kill criterion came first. That is
  the part of the idea this result does not touch, but it is close to what
  Memento and others already do.
- All of this is synthetic. No real Ada precedents.

## Runs that did not count

Two runs on Qwen3-0.6B-4bit are void and kept in `void_runs/`. Run 1 used six
fixed decision pairs and invented words: every arm memorised the training labels
and `move` emitted one constant string. Run 2 fixed the words (fresh single-token
English words each example, train and test pools disjoint) and the model still
collapsed; `move` output gibberish and scored 0%. The move to 1.7B and a lower
learning rate came after seeing those failures, and only the data and model
changed. The bar and the arms did not.
