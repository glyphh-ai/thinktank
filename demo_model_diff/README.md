# demo_model_diff — a semantic changelog for fine-tuning

**What did instruction tuning do to meaning?** This demo answers with named
numbers: the same 1375-word vocabulary is read through Qwen2.5-0.5B (base)
and Qwen2.5-0.5B-Instruct on the 242 named contrast axes from
`version_12_wide_seed`, each model measured in its own frame, and the
readings diffed.

The output (`REPORT.md`) is a changelog no eval suite produces:

- which named semantic dimensions instruction tuning disturbed vs left alone
- which words moved most, overall and on axes you'd ask about
  (good–bad, polite–rude, safe–dangerous, obey–rebel, ...)
- discrete **bit flips**: words whose committed reading reversed sign

This is the origin idea of the whole research line — *"take some algebra of
the differences between models"* — landed in the space where versions 07–12
showed meaning is actually measurable. It generalizes to any pair of
checkpoints sharing a tokenizer-compatible extraction: base vs fine-tune,
pre vs post safety training, week-over-week continual training.

Word-level caveat: readings use the raw-text carrier frames (no chat
template), so this measures representational drift, not chat behavior.

## Result (see REPORT.md)

Instruction tuning at 0.5B is **representation-preserving at the word
level**: median axis correlation base↔instruct **0.993**, and **zero**
committed bit flips across 1375×242 bits — whatever chat tuning changed, it
did not move word meanings wholesale. Against that quiet baseline the
detector's top signal is unmistakably real: the single most-moved word is
**"mean"** (‖Δz‖ 4.0, 1.5× the runner-up), shifting toward *bad*, *rude*,
*angry*, *false*, and *dishonest* — the chat tune re-weighted its dominant
sense from statistics ("the mean") to unkindness ("that's mean"). The next
movers are chat-register words (*reply, answer, issue, date, update*), and
*update/edit/help/useful* drifted toward *good/polite* — assistant-speak
acquiring positive valence. Small, coherent, and named: exactly what a
semantic changelog is for.

## Run

```
python extract.py Qwen/Qwen2.5-0.5B-Instruct instruct_vectors.npz  # ~30 min CPU
python diff.py                                                      # writes REPORT.md
```
