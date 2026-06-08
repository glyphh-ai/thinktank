"""A suite of causal plot holes (and decoys) spread across >1M tokens.

One ~1.6M-token document carrying ~10 genuine plot holes -- a permanent state
established early and violated far later, with no event to explain it -- and ~5
decoys that look identical but ARE explained by an event in between (and so must
NOT be flagged). Many setup/violation pairs are >1M tokens apart, so no single
LLM call can see both halves: only chunk-extract -> world-model -> holistic
review can detect them, and only by tracking the licensing event can it avoid the
decoys. One read (~$8-10) yields a real recall/precision distribution.
"""

from __future__ import annotations

from app.text import word_count

from .cases import Case, Decoy, Plant
from .corpus import _FILLER_POOL

# (entity, early=permanent-state, late=violation, keywords)
_PLOT_HOLES = [
    ("Aurelian vault",
     "The Aurelian vault could never be opened again: its only key was melted to slag and the lock welded shut for all time.",
     "Mara drew the Aurelian key, turned it in the lock, and the Aurelian vault swung open.",
     ["aurelian", "vault", "open", "seal", "weld", "melt"]),
    ("Cartwright map",
     "The only Cartwright map was burned to ash in the hearth, gone beyond recovery.",
     "She unrolled the Cartwright map and traced the route with her finger.",
     ["cartwright", "map", "burn", "ash"]),
    ("Thornwood bridge",
     "The Thornwood bridge collapsed into the gorge and was never rebuilt.",
     "They crossed the Thornwood bridge at dawn without breaking stride.",
     ["thornwood", "bridge", "collaps", "cross"]),
    ("Vesper antidote",
     "The last vial of the Vesper antidote shattered on the stones, the formula lost forever.",
     "The physician administered the Vesper antidote and the fever broke.",
     ["vesper", "antidote", "shatter", "administ"]),
    ("Tobias",
     "Tobias died of his wounds that night and was buried beneath the chapel floor.",
     "Tobias strolled into the tavern, called for ale, and laughed at the news.",
     ["tobias", "died", "buried", "alive", "walk"]),
    ("Marigold",
     "The Marigold struck the reef and sank to the seabed with all her timbers.",
     "They boarded the Marigold and sailed her out on the morning tide.",
     ["marigold", "sank", "sail", "ship"]),
    ("Sefton key",
     "The Sefton key was hurled from the cliff into the deep sea, never to be found.",
     "He produced the Sefton key and unlocked the iron door.",
     ["sefton", "key", "sea", "unlock"]),
    ("Halloran tower",
     "The Halloran tower was demolished to rubble, not one stone left standing.",
     "She climbed the spiral stair of the Halloran tower to the very top.",
     ["halloran", "tower", "demolish", "climb", "rubble"]),
    ("Pell letter",
     "The Pell letter was thrown unread into the fire and consumed entirely.",
     "He quoted the Pell letter word for word to the assembled court.",
     ["pell", "letter", "burn", "unread", "quote"]),
    ("Karelian army",
     "The Karelian army was disbanded forever, its banners burned and soldiers sent home.",
     "The Karelian army massed at the border and marched at the king's command.",
     ["karelian", "army", "disband", "march"]),
]

# (entity, early=state, event=licenses the change, late=use, keywords)
_DECOYS = [
    ("Brevik lighthouse",
     "The Brevik lighthouse went dark when its great lamp shattered in the storm.",
     "A new lamp was shipped in and installed in the Brevik lighthouse that month.",
     "The Brevik lighthouse shone bright across the bay once more.",
     ["brevik", "lighthouse", "dark", "shone", "lamp"]),
    ("Della",
     "Word came that Della had died in the crossing, and the house went into mourning.",
     "In truth the death was a ruse Della had staged to escape her creditors.",
     "Della returned in spring, very much alive, to reclaim her estate.",
     ["della", "died", "alive", "return", "ruse"]),
    ("Mossgate road",
     "The Mossgate road washed away entirely in the flood.",
     "Crews rebuilt the Mossgate road over the summer, stronger than before.",
     "Carts rolled down the Mossgate road again by autumn.",
     ["mossgate", "road", "wash", "rebuilt"]),
    ("Harrow well",
     "The Harrow well ran completely dry in the long drought.",
     "The autumn rains returned and refilled the Harrow well to the brim.",
     "They drew cold water from the Harrow well each morning.",
     ["harrow", "well", "dry", "water"]),
    ("Ferro clock",
     "The Ferro clock stopped dead, its mainspring snapped beyond repair.",
     "A clockmaker fitted a new mainspring and set the Ferro clock running again.",
     "The Ferro clock chimed the hour across the square.",
     ["ferro", "clock", "stop", "chime", "repair"]),
]


def build_xl_corpus(target_words: int = 950_000, seed: int = 11) -> Case:
    signals: list[tuple[float, str]] = []
    plants: list[Plant] = []
    decoys: list[Decoy] = []

    span = max(1, len(_PLOT_HOLES) - 1)
    for i, (ent, early, late, kw) in enumerate(_PLOT_HOLES):
        e = 0.02 + 0.14 * (i / span)   # setup in the first ~16%
        l = 0.85 + 0.13 * (i / span)   # violation in the last ~15% (>1M tokens later)
        signals += [(e, early), (l, late)]
        plants.append(Plant(ent, "status", "continuity", keywords=kw))

    for i, (ent, early, event, late, kw) in enumerate(_DECOYS):
        e = 0.03 + 0.12 * (i / 4)
        m = 0.45 + 0.08 * (i / 4)      # licensing event in the middle
        l = 0.86 + 0.12 * (i / 4)
        signals += [(e, early), (m, event), (l, late)]
        decoys.append(Decoy(ent, "status", keywords=kw))

    per = word_count(" ".join(_FILLER_POOL[:4]))
    n = max(1000, target_words // per)
    paras = [" ".join(_FILLER_POOL[(i + k) % len(_FILLER_POOL)] for k in range(4))
             + f" (ledger entry {i})." for i in range(n)]
    for frac, line in sorted(signals, key=lambda s: s[0], reverse=True):
        paras.insert(min(int(frac * n), len(paras)), line)

    return Case(id="xl_suite", text="\n\n".join(paras), plants=plants, decoys=decoys)
