"""A real, sizable benchmark corpus with a precise answer key.

Generates one long document (~18k words by default) seeded with many planted
inconsistencies across varied types and many entities, plus decoys (changes an
in-story event legitimately explains, which must NOT be flagged). Deterministic.

This is a controlled ground truth -- the honest way to score recall/precision
when no public answer key exists. ~30 plants + ~12 decoys so a single miss moves
recall ~3%, not 20%.
"""

from __future__ import annotations

import random

from app.text import word_count

from .cases import Case, Decoy, Plant

# Distinct first names so coreference is unambiguous across the document.
_NAMES = [
    "Elena", "Marcus", "Quinn", "Iris", "Hale", "Reyes", "Dalton", "Mira", "Cole",
    "Selene", "Ravi", "Nadia", "Ottoline", "Bram", "Yusuf", "Petra", "Lars", "Greta",
    "Tomas", "Vivian", "Soren", "Adaeze", "Kiran", "Magnus", "Priya", "Felix", "Rosa",
    "Dexter", "Wren", "Anton", "Lucia", "Hugo", "Della", "Ivo", "Maren", "Cyrus",
    "Noor", "Edmund", "Talia", "Gideon",
]

# Plant templates: (key, category, early(name)->str, late(name)->str)
_PLANTS = [
    ("eye_color", "continuity",
     lambda n: f"{n}'s eyes were a clear green that morning.",
     lambda n: f"In the lamplight {n}'s eyes were a deep brown."),
    ("age", "timeline",
     lambda n: f"{n} was 34 years old when the contract was signed.",
     lambda n: f"The clerk recorded that {n} was 52 years old."),
    ("role", "character",
     lambda n: f"{n}, the ship's surgeon, prepared the instruments.",
     lambda n: f"{n}, a tax accountant by trade, reviewed the ledgers."),
    ("height", "continuity",
     lambda n: f"{n} was the tallest in the room, a full head above the rest.",
     lambda n: f"{n}, the shortest of them, had to stand on a crate to be seen."),
    ("hometown", "knowledge",
     lambda n: f"{n} had never once left the village of their birth.",
     lambda n: f"{n} spoke of the decade spent abroad in distant cities."),
    ("object_status", "continuity",
     lambda n: f"{n}'s pocket-watch was crushed beyond repair under the cart.",
     lambda n: f"{n} checked the time on the same pocket-watch and smiled."),
    ("sibling_count", "continuity",
     lambda n: f"{n} was an only child, or so everyone believed.",
     lambda n: f"{n} introduced all three of their younger sisters."),
]

# Decoy templates: (key, early, event(licenses the change), late)
_DECOYS = [
    ("hair", lambda n: f"{n} had jet-black hair when they set out.",
     lambda n: f"{n} dyed their hair blonde at a roadside inn to avoid notice.",
     lambda n: f"{n}, now blonde, kept their hood low."),
    ("rank", lambda n: f"Lieutenant {n} saluted the colours at dawn.",
     lambda n: f"{n} was promoted to captain for valour during the siege.",
     lambda n: f"Captain {n} gave the final order at dusk."),
    ("residence", lambda n: f"{n} lived for years in the river district.",
     lambda n: f"{n} packed up and moved across town to the hill quarter.",
     lambda n: f"{n} now lived in the hill quarter, settled at last."),
]

_FILLER_POOL = [
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


def build_corpus(num_plants: int = 28, num_decoys: int = 9, target_words: int = 18_000,
                 seed: int = 7) -> Case:
    rng = random.Random(seed)
    names = _NAMES.copy()
    rng.shuffle(names)
    name_iter = iter(names)

    # (fraction, sentence) signals to weave through the document.
    signals: list[tuple[float, str]] = []
    plants: list[Plant] = []
    decoys: list[Decoy] = []

    for i in range(num_plants):
        key, cat, early, late = _PLANTS[i % len(_PLANTS)]
        n = next(name_iter)
        e = 0.02 + 0.30 * rng.random()   # early state in the first third
        l = 0.66 + 0.32 * rng.random()   # late state in the last third
        signals += [(e, early(n)), (l, late(n))]
        plants.append(Plant(n, key, cat))

    for i in range(num_decoys):
        key, early, event, late = _DECOYS[i % len(_DECOYS)]
        n = next(name_iter)
        e = 0.02 + 0.28 * rng.random()
        m = 0.40 + 0.20 * rng.random()   # licensing event in the middle
        l = 0.70 + 0.28 * rng.random()
        signals += [(e, early(n)), (m, event(n)), (l, late(n))]
        decoys.append(Decoy(n, key))

    per = word_count(" ".join(_FILLER_POOL[:4]))
    n_para = max(120, target_words // per)
    paras = [" ".join(_FILLER_POOL[(i + k) % len(_FILLER_POOL)] for k in range(4))
             for i in range(n_para)]
    for frac, line in sorted(signals, key=lambda s: s[0], reverse=True):
        paras.insert(min(int(frac * n_para), len(paras)), line)

    return Case(id="benchmark", text="\n\n".join(paras), plants=plants, decoys=decoys)
