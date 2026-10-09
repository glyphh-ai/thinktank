# Curriculum — summary of the three runs

Organic: same rule, schedule fixed before running, exams never trained on.

| run | stage | human Q top-1 / top-5 / top-20 | human median rank | held-out top-1 / top-5 / median | trusts | mutes |
|---|---|---|---|---|---|---|
| real speech (main) | newborn | 0.0% / 6.2% / 10.4% | 160 | - | class, base, number | time, cause, purpose |
| real speech (main) | baby | 2.1% / 6.2% / 16.7% | 177 | 0.0% / 0.6% / 31321 | strength, time, place | after, opposite, similar |
| real speech (main) | toddler | 4.2% / 8.3% / 14.6% | 131 | 0.0% / 0.0% / 28092 | time, strength, category | opposite, after, similar |
| real speech (main) | child | 4.2% / 12.5% / 20.8% | 87 | 0.0% / 0.3% / 31363 | strength, category, number | partner, phrase, opposite |
| real speech (main) | school | 4.2% / 12.5% / 20.8% | 88 | 2.0% / 4.3% / 7053 | strength, number, register | figure, phrase, opposite |
| real speech (replicate) | newborn | 0.0% / 6.2% / 10.4% | 160 | - | class, base, number | time, cause, purpose |
| real speech (replicate) | baby | 0.0% / 6.2% / 14.6% | 184 | 0.0% / 0.6% / 31728 | time, strength, place | opposite, after, similar |
| real speech (replicate) | toddler | 4.2% / 8.3% / 16.7% | 129 | 0.3% / 1.0% / 32226 | time, strength, number | opposite, after, similar |
| real speech (replicate) | child | 4.2% / 12.5% / 18.8% | 88 | 0.0% / 0.7% / 27726 | strength, number, register | partner, opposite, similar |
| real speech (replicate) | school | 4.2% / 12.5% / 22.9% | 88 | 1.3% / 4.7% / 5585 | strength, register, number | figure, opposite, similar |
| shuffled words (control) | newborn | 0.0% / 6.2% / 10.4% | 160 | - | class, base, number | time, cause, purpose |
| shuffled words (control) | baby | 2.1% / 4.2% / 18.8% | 140 | 0.0% / 0.6% / 30548 | strength, feeling, time | after, opposite, similar |
| shuffled words (control) | toddler | 4.2% / 8.3% / 16.7% | 100 | 0.0% / 0.0% / 27688 | strength, feeling, register | similar, opposite, after |
| shuffled words (control) | child | 4.2% / 12.5% / 18.8% | 86 | 0.0% / 0.3% / 30858 | strength, feeling, register | partner, phrase, opposite |
| shuffled words (control) | school | 4.2% / 12.5% / 20.8% | 100 | 2.3% / 4.3% / 9508 | strength, register, number | similar, opposite, purpose |

**Finding:** the brain improves steadily from newborn to school on the human questions
(median rank 160 -> 87, top-5 6.2% -> 12.5%) and the replicate reproduces it — but the
shuffled-word control improves exactly as much. What it learns is exposure to everyday
vocabulary and which kinds of threads to trust overall, not the structure of language.
Held-out prediction is at chance until school (definitions), where all three reach
~2% top-1 / median rank ~7,000 — again with the control matching.
