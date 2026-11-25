# FINDINGS — version_23_lottery

Rules, split and pass bar: `PROTOCOL.md` (written before each run).
Logs: `run.log` (run 1), `run_hdc.log` (Addendum A). Numbers: `results.json`,
`results_hdc.json`. Data: Texas Lottery CSVs through 2026-11-21.

## The answer: no pattern, and this time the "no" is trustworthy

The HDC model with beam-search tickets **found the planted pattern** (its
tickets won 22 times against 5.3 expected, p 4e-8; its top 10 tickets won
194 times against 53) and **correctly found nothing in the random stream**.
Run through the same pipeline, it found nothing in any real Texas game.

| game (test draws) | HDC gain, bits/draw | ticket vs chance | verdict |
|---|---|---|---|
| control: planted (5,264) | **+0.165** (p 2e-38) | **22 vs 5.3** | **PASS** (as it must) |
| control: random (5,264) | 0.000 (chose β = 0) | 3 vs 5.3 | fail (as it must) |
| Lotto Texas 6/54 (484) | −0.003 | 294 vs 323 balls | fail |
| Cash Five 5/35 (512) | 0.000 (chose β = 0) | 356 vs 366 balls | fail |
| Texas Two Step (533) | −0.001 | 248 vs 244 balls | fail |
| Pick 3 (5,264) | −0.000 | 10 vs 5.3 straight wins (p 0.04) | fail |
| Daily 4 (4,052) | 0.000 (chose β = 0) | 0 vs 0.4 | fail |

On validation, HDC picked β = 0 (uniform: "I know nothing") for Cash Five,
Daily 4 and the random stream, and β ≤ 0.03 everywhere else. Pick 3's 10
wins is the closest thing to a signal, at p 0.04, 80× short of the 0.0005
bar, with no gain in bits, so it's luck.

## Run 1: every model failed, but two of them failed the planted pattern too

| model | real games passed | planted pattern | read |
|---|---|---|---|
| hot numbers | 0 / 5 | fail (by design: can't see order) | the folk theory, nothing there |
| bitspin (v21, default heads) | 0 / 5 | fail (−4.0 bits/draw) | overconfident on random data: 2.4–4.8 bits/draw worse than uniform everywhere, so its "no" says nothing |
| byte transformer (v21 ByteLM, small) | 0 / 5 | fail (−0.006) | didn't learn a 2-byte-back copy in 5,600 steps; its "no" says nothing |

Bitspin's tickets beat chance on Lotto (p 0.012) and Cash Five (p 0.0055)
but not Two Step (p 0.75) or the digit games; with no gain in bits and the
bar at 0.0007, that's noise. No duplicate draws in the data (Pick 3's 29
back-to-back repeats are about the 26 chance expects).

## Not tried

The fireball digits; games before their current rules; any other model
setting after seeing a test.
