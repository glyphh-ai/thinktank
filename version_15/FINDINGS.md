# FINDINGS — version_15

## The ground wire — dead end

v14 proved 3-phase π-frequency attention captures 75% of multi-head's
advantage through geometric position coverage. Lightweight adapters
couldn't close the remaining 25%. v15 asked: can a recurrent memory
signal (the "ground wire") close it instead?

### Design

Two-pass attention inspired by the neutral wire in 3-phase electrical
systems:

  Pass 1 → standard 3-phase attention (no ground)
  Ground → causal EMA of pass 1 output (memory of what's been attended)
  Pass 2 → 3-phase with Q += α·ground (queries steered by memory)

Two learned scalars: ground_alpha (injection strength), ground_decay
(EMA rate via sigmoid). Total new params: 2.

### Results

| model                          | params    | val loss | val acc |
|--------------------------------|-----------|----------|---------|
| v0 baseline (1-head, learned)  | 1,579,391 | 3.12     | 35.1%   |
| v14 3-phase π (no ground)      | 1,579,391 | 3.08     | 36.8%   |
| **v15 3-phase π + ground**     | 1,579,393 | **3.086**| **36.6%** |
| v14 3-head multi-head          | 1,582,463 | 2.999    | 37.8%   |
| **v15 12-head multi-head**     | 1,579,391 | **2.998**| **37.4%** |

Ground wire: **no improvement.** Within noise of the no-ground model.

### Learned parameters tell the story

The model tried hard to use the ground:
- alpha: 0.1 → **0.76** (7.6× increase — it WANTS the signal)
- decay: 0.88 → **0.45** (collapsed — only cares about ~2 positions back)

It cranked up the injection strength but killed the memory length.
The two-pass approximation is too weak — the ground is computed from
pass 1 (without ground), so it's always one step behind the model's
actual behavior. And attention already provides causal memory through
the mask — an explicit EMA is redundant with what the model already has
full access to.

### 12-head comparison

12 heads (head_dim=10) performs identically to 3 heads (head_dim=42):
2.998 vs 2.999 val loss. At C=128, 3 heads already saturates content
specialization — more heads with smaller subspaces adds nothing.

### Verdict

The ground wire is dead. The remaining gap to multi-head is subspace
projection — each head seeing a different slice of the embedding space.
Not memory, not adapters, not phase count, not attention steering.

## Reproduce

```
python test_train.py                       # gradient check + 700-step gate
python -u strat0_train.py                  # ground wire 8k run
python -u strat1_12head_train.py           # 12-head comparison
```
