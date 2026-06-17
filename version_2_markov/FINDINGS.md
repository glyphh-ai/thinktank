# version_2_markov — findings

Question: does choosing training windows by a Markov-style priority (sample
where the model is currently weak) beat uniform random sampling?

All runs: char-level GPT, Adam, tiny-Shakespeare corpus (1.1M chars, vocab 65).

## 1. Base model (C=64, ctx=16), 2000 steps, 3 seeds — `sweep_sampling.py`

```
config                val loss (mean±std)   val acc   ->match uniform @
uniform                 2.085 ± 0.011         40.4%      2000 (ref)
prio b=.9  e=.10        2.086 ± 0.001         40.2%      1767
prio b=.99 e=.10        2.106 ± 0.012         39.6%      1800
prio b=.9  e=.05        2.086 ± 0.013         40.4%      1733
prio b=.9  e=.20        2.094 ± 0.007         40.5%      1800
```

- **No final-quality gain**: prioritized final val loss is inside uniform's error
  bar. (An earlier 2-seed run that looked like a win was sampling noise.)
- **Modest convergence-speed gain**: prioritized reaches uniform's *final* loss
  ~10-13% sooner (best: b=.9/e=.05 at 1733 vs 2000 steps).
- Tuning: beta=0.9 > beta=0.99 (long memory -> stale priorities hurt);
  eps in 0.05-0.10 is best.

## 2. Scaled model (C=128, ctx=32, ~219k params), 1500 steps, 2 seeds — `compare_scaled.py`

```
              final val loss  final val acc  best val loss
uniform            2.037          41.1%          1.981
prioritized        2.072          39.7%          2.002
```

- Prioritized is **slightly worse** and trails uniform across the whole curve.
  The small-scale speed advantage does not survive scaling up.

## Verdict

Prioritized ("Markov-over-positions") sampling is **not a clear win**. At small
scale it's a minor convergence-speed optimization with no quality gain; at larger
scale it slightly *underperforms* uniform.

Most likely cause: sampling non-uniformly **biases the training distribution**
away from the true one, so the model optimizes a reweighted objective that
doesn't match the (uniform) validation distribution. Prioritized experience
replay hits the same problem and fixes it with **importance-sampling weights**
(scale each sample's gradient by ~1/priority) to keep the gradient unbiased.
We did NOT do that correction here -> the prime suspect for why it doesn't help.

## If continuing
- Add IS-weight correction to `PrioritizedSampler` (the principled fix).
- Re-tune beta/eps at the scaled size (they were only tuned at base size).
- More seeds on the scaled model (only 2 here).
