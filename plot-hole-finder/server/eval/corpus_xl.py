"""A plot hole that spans more than 1,000,000 tokens.

A hard causal constraint is established near the start -- the Aurelian vault is
sealed for all time, its only key melted to slag, the lock welded shut -- and
violated ~1M tokens later, when someone opens that vault with that key. The two
halves are deliberately more than a 1M-token context window apart, so no single
LLM call can see both: the baseline physically cannot detect it. Only
chunk-extract -> compressed-world-model -> holistic review can.
"""

from __future__ import annotations

from app.text import word_count

from .cases import Case, Plant
from .corpus import _FILLER_POOL

_SETUP = (
    "The Aurelian vault could never be opened again. Its only key was melted to "
    "slag in the foundry furnace, the great lock was welded shut with iron, and "
    "the door was sealed for all time -- no one alive could ever open it."
)
_VIOLATION = (
    "Mara drew the Aurelian key from her coat, turned it in the welded lock, and "
    "the Aurelian vault swung open without complaint as she stepped inside."
)


def build_xl_corpus(target_words: int = 950_000, seed: int = 11) -> Case:
    # ~950k words, comfortably past the 1M-token context window.
    per = word_count(" ".join(_FILLER_POOL[:4]))
    n = max(1000, target_words // per)
    # Unique suffix per paragraph so no two chunks are identical (avoids a
    # degenerate-input refusal on huge repetitive text).
    paras = [" ".join(_FILLER_POOL[(i + k) % len(_FILLER_POOL)] for k in range(4))
             + f" (ledger entry {i})." for i in range(n)]
    paras.insert(int(0.01 * n), _SETUP)   # constraint established near the start
    paras.append(_VIOLATION)              # constraint violated at the very end

    return Case(
        id="xl_plot_hole",
        text="\n\n".join(paras),
        plants=[Plant("Aurelian", "status", "continuity",
                      keywords=["seal", "weld", "melt", "open", "vault", "key"])],
    )
