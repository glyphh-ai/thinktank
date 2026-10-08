# FINDINGS — version_17_ada

Glyphs (Haiku-written, closed set, identical 27-role makeup) + a brain made of
the lattice's own layers. All numbers on words never used for training.

| file | finding |
|---|---|
| `typed_encode.py` | typed bindings: decode 98.0% -> 99.9%; synonyms love/care .37 -> .58; opposites stay positive (hate's own glyph pulls toward love) |
| `context_fire.py` | one glyph per word; context alone splits "mean" into cruel (hurt/unkind/kind) vs intend (verb/word/say) |
| `reinforce.py` | same structure + rule, different childhood: loved child separates love/hate (0.59 -> 0.02), abused child's love takes hate's valence (-1) |
| `think.py` | linear settling: exact local learning (EP = gradient, cosine 1.0000) but retrieval riddles: plain lookup 70.7% >> settling 25.8%; per-thread weights memorize |
| `reason.py` | second-order riddles: linear settling 0% — a sum of fields is an OR; coincidence detection is the missing AND |
| `brain.py` | layers = regions, directed flow (value -> holder, energy split), only this step's firing, one stream per clue, coincidence: top-1 7.7% vs explicit 2-step search 2.1%; 100% where search is unique |
| `flow.py` | mixed-depth riddles (clues 1-3 hops). Explicit search ≤3 hops 0.3% top-1. Brain told depth 1/2/3: 29.6/28.9/20.0%; **finds its own depth: 26.3%** (stops at 3.1 steps). **Learning pathway gains** (exact local gradient = forward activity x backward error, cosine 1.000000): unseen words **26.3% -> 34.4% top-1, 43.8% -> 59.2% top-5, median rank 7 -> 2** |

Learned regional gains (flow.py): grammar 3.75, idiom 1.58, sentence 1.02,
meaning 0.68 — it trusts grammar pathways (number, degree, class) and mutes
partner/phrase/figure/similar. In this flow a grammar clue ("noun", "some")
reaches every holder weakly, so it works as a broad constraint that the
coincidence step sharpens.
