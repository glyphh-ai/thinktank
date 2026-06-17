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

## 3. Importance-sampling correction (C=64, ctx=16), 2000 steps, 3 seeds — `compare_is.py`

The principled fix: scale each window's gradient by w_i=(1/M)/p_i so the
expected gradient equals the uniform objective (as in prioritized experience
replay). Gradient-checked (weighted path rel err ~1e-7).

```
sampler          val loss (mean±std)   val acc   ->match uniform @
uniform            2.085 ± 0.011        40.4%      2000 (ref)
prio (no IS)       2.086 ± 0.001        40.2%      1767
prio (IS-corr)     2.097 ± 0.015        40.5%      1800   (highest variance)
```

IS correction does NOT rescue it. All three tie on final val loss (within error
bars), and the corrected version has the highest variance and is marginally
slowest.

## Verdict (final)

Prioritized ("Markov-over-positions") sampling is **not a win** for this
stationary supervised LM objective. The reason is a genuine bias-variance trap:

- **Uncorrected** -> the sampling skew BIASES the gradient (optimizes a
  reweighted objective != validation) -> no net gain; mild harm when scaled up.
- **IS-corrected** -> the gradient expectation becomes IDENTICAL to uniform by
  construction, so the ceiling is "tie uniform" -- but the reweighting INFLATES
  gradient variance, landing at "uniform, but noisier."

There is no free lunch here: prioritization either biases you or just adds
variance. It pays off in the settings it was designed for -- non-stationary
data, sparse rewards, replay buffers (RL) -- not on a stationary objective where
uniform sampling already covers the data well.

## Arc summary
- v1 (Markov optimizer): tied/lost to Adam -- Adam's 2nd moment already encodes
  the oscillation signal.
- v2 (Markov sampling): no win; speed-only at small scale, mild loss at scale,
  and IS correction confirms the bias-variance trap rather than removing it.

Two clean negative results, each with an understood mechanism.
