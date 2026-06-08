"""Labeled evaluation cases: documents with known plot holes (and decoys).

Each case carries the ground truth:
- ``plants``  -- inconsistencies that SHOULD be flagged (drives recall)
- ``decoys``  -- legitimate changes that should NOT be flagged (a licensed hair
                 dye, etc.) -- flagging one is a false positive (drives precision)

Cases marked ``llm_only`` need real LLM extraction (subtle/semantic) and are
skipped when the offline heuristic backend is active.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Plant:
    entity: str
    key: str        # attribute that contradicts, e.g. "eye_color"
    category: str = "continuity"


@dataclass
class Decoy:
    entity: str
    key: str        # a change that an in-story event licenses -> must not flag


@dataclass
class Case:
    id: str
    text: str
    plants: list[Plant] = field(default_factory=list)
    decoys: list[Decoy] = field(default_factory=list)
    llm_only: bool = False


def _doc(*paras: str) -> str:
    filler = ("The hall was quiet but for the wind against the shutters, and the "
              "business of the day carried on as it always had. ")
    out = []
    for p in paras:
        out.append(p)
        out.append(filler)  # push assertions into separate passages
    return "\n\n".join(out)


CASES: list[Case] = [
    Case(
        id="eye_color_continuity",
        text=_doc(
            "Elena's eyes were green when she first arrived at the keep.",
            "She crossed the courtyard and climbed the long stair to the tower.",
            "In the final chamber, Elena's eyes were blue, though no one remarked on it.",
        ),
        plants=[Plant("Elena", "eye_color", "continuity")],
    ),
    Case(
        id="licensed_hair_change",
        text=_doc(
            "Marcus had black hair when the journey began.",
            "At a roadside inn, Marcus dyed his hair blonde to avoid recognition.",
            "By the time they reached the city, Marcus had blonde hair.",
        ),
        decoys=[Decoy("Marcus", "hair")],  # licensed by the dye event -> no flag
    ),
    Case(
        id="age_timeline",
        text=_doc(
            "Senator Hale was 60 years old when he first took the oath of office.",
            "The session dragged on through the long winter of debate.",
            "By the closing statement, Senator Hale was 40 years old per the record.",
        ),
        plants=[Plant("Hale", "age", "timeline")],
    ),
    Case(
        id="clean_control",
        text=_doc(
            "Iris had green eyes, the colour of river glass.",
            "She worked the docks from dawn, hauling crates with the others.",
            "Even at dusk, Iris had green eyes that caught the lantern light.",
        ),
        # No plants, no decoys: a clean document. Flagging anything here is a
        # precision failure.
    ),
    Case(
        id="role_contradiction_llm",
        text=_doc(
            "Dr. Quinn, the cardiac surgeon, scrubbed in before the long operation.",
            "The hospital corridors emptied as the night shift took over.",
            "Later, Quinn, a defense attorney, addressed the jury with practiced calm.",
        ),
        plants=[Plant("Quinn", "role", "character")],
        llm_only=True,  # heuristic has no 'role' pattern; needs LLM extraction
    ),
]
