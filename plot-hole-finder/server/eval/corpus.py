"""A substantial benchmark document with a precise answer key.

One ~10k-word document carrying inconsistencies of several kinds spread far
apart, plus decoys (changes an in-story event legitimately explains). This is a
controlled ground truth -- the only way to score recall/precision honestly when
no public answer key exists. Difficulty is mixed on purpose:

- structured-friendly: eye colour, age (clean attribute slots)
- holistic-only:       role/profession flip, an object destroyed then worn, a
                       character's knowledge contradicting itself
- decoys:              a dyed haircut and a military promotion (must NOT flag)
"""

from __future__ import annotations

from app.text import word_count

from .cases import Case, Decoy, Plant

# (fraction-through-document, sentence). Early states first, late states last,
# licensing events for decoys in the middle.
_SIGNALS = [
    (0.04, "Elena's eyes were a clear green that morning as she reviewed the documents."),
    (0.07, "Senator Hale was 60 years old when he first swore the oath of office."),
    (0.10, "Dr. Quinn, the cardiac surgeon, scrubbed in for the long operation."),
    (0.13, "Iris admitted quietly that she had never once set foot in the capital."),
    (0.16, "Marcus had black hair when the company set out from the harbour."),
    (0.19, "Lieutenant Reyes saluted the colours at dawn before the march."),

    (0.46, "The silver amulet shattered into dust and was lost forever in the fire."),
    (0.50, "At a roadside inn, Marcus dyed his hair blonde to avoid being recognised."),
    (0.54, "Reyes was promoted to captain for valour during the long siege."),

    (0.80, "In the lamplight Elena's eyes were blue, though the record never explained it."),
    (0.83, "By the final hearing the clerk noted that Senator Hale was 40 years old."),
    (0.86, "Quinn, a seasoned defense attorney, rose to address the jury with calm."),
    (0.89, "Iris reminisced fondly about the long years she had lived in the capital."),
    (0.92, "She clasped the silver amulet at her throat as she entered the great hall."),
    (0.95, "Marcus, blonde now, kept his hood low as they passed the gate."),
    (0.97, "Captain Reyes gave the final order at dusk."),
]

# A pool of distinct neutral sentences so the filler reads like real prose rather
# than one paragraph repeated (which is degenerate and can trip a false refusal).
_POOL = [
    "The work of the day carried on much as it always had, papers passing from hand to hand.",
    "Outside, the city went about its business under a flat grey sky.",
    "Talk in the corridors turned, as it often did, to the weather and the price of bread.",
    "A clerk stacked ledgers by the window and trimmed the lamp against the gloom.",
    "Carts rattled over the cobbles below, and a bell tolled somewhere across the river.",
    "The long afternoon wore on toward dark with little of consequence decided.",
    "Messengers came and went, their boots loud on the worn stone stairs.",
    "Rain threatened but never quite fell, and the gutters stayed dry.",
    "Someone laughed in a far room; a door closed; the quiet settled again.",
    "The kettle was set to boil and the morning's correspondence sorted into trays.",
    "A draught stirred the curtains and guttered the candles along the hall.",
    "The guards changed at the gate without ceremony as the watch turned over.",
]


def _filler(i: int) -> str:
    return " ".join(_POOL[(i + k) % len(_POOL)] for k in range(4))


def build_corpus(target_words: int = 10_000) -> Case:
    per = word_count(_filler(0))
    n = max(60, target_words // per)
    paras: list[str] = [_filler(i) for i in range(n)]
    # Insert signal sentences at their fractional positions (back-to-front so
    # earlier indices stay valid).
    for frac, line in sorted(_SIGNALS, key=lambda s: s[0], reverse=True):
        paras.insert(int(frac * n), line)
    text = "\n\n".join(paras)

    return Case(
        id="benchmark_10k",
        text=text,
        plants=[
            Plant("Elena", "eye_color", "continuity"),
            Plant("Hale", "age", "timeline"),
            Plant("Quinn", "role", "character"),
            Plant("amulet", "status", "continuity"),
            Plant("Iris", "capital", "knowledge"),
        ],
        decoys=[Decoy("Marcus", "hair"), Decoy("Reyes", "rank")],
    )
