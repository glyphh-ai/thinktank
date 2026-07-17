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

## Run

```
python extract.py Qwen/Qwen2.5-0.5B-Instruct instruct_vectors.npz  # ~30 min CPU
python diff.py                                                      # writes REPORT.md
```
