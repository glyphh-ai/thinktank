# Meaning compounding — the decisive result

Question: do **selectional frames** (a verb's argument restrictions) taught from
one corpus transfer to the next — the meaning-equivalent of vocabulary's 91%?

## Setup
- 12 verbs common to both corpora (said, know, come, think, went, see, get, …).
- Real agents extracted each occurrence's (subject-class, object-class) over a
  fixed 9-class set (ANIMATE, PHYSICAL, PLACE, ABSTRACT, COMMUNICATION, …).
- Corpus 1 (Alice) → frame per verb = the set of arg signatures seen.
- Corpus 2 (Looking-Glass) → test each occurrence against the corpus-1 frame.

## Result
| measure | value |
|---|---|
| verb frame already known | **48/48 = 100%** |
| exact frame match (subj+obj) | 23/48 = 48% |
| **assimilated** (subject-class fits) | **43/48 = 90%** |
| needed accommodation (new frame) | 5/48 = 10% |

**Meaning compounds at ~90% — essentially the same as vocabulary (91%).** The
selectional structure of common verbs is stable across corpora: corpus-1 frames
absorb 90% of corpus-2 usage. The 10% that don't are real sense shifts
("her paw *went* into your eye" = BODY; "*got* cold" = ABSTRACT) — exactly the
cases `evolve.py`'s consolidation/accommodation handles.

## Honest caveats (90% is likely optimistic)
- common verbs only (most stable case, like the 91% was frequent words);
- coarse 9-class semantics (fine restrictions would vary more);
- same author/domain (cross-domain would stress it);
- small sample (48 occurrences).
Cross-domain + rare verbs + finer classes would lower it — but even a drop to
~70% is still compounding, with accommodation handling the remainder.

## Why it matters
This was the open question that decides the thesis. The semantic-frame tower
looks **tractable and compounding**, not the unbounded LLM-scale problem. Meaning
appears to behave like vocabulary: taught once, reused cheaply. First strong
evidence that "grounded understanding, built by a swarm, compounding across
corpora" is real for the layer that was most in doubt.
