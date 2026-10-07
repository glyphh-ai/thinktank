"""
THE GLYPH SCHEMA — one fixed lattice for every word
===================================================
layer > segment > role, the same 21 roles for every word. A word's glyph
fills each role with ONE word (or "none") and a weight 0..1. Role values are
open vocabulary: any value word must itself get a glyph (see closure.py), so
the set of glyphs is closed — every word used to describe a word is described.

This file is pure data + validation. No API, no vectors.
"""

import re

# layer -> {segment -> {role -> what the role asks}}
SCHEMA = {
    "being": {
        "what": {
            "category": "what kind of thing it is (feeling, animal, tool, action, place, color, ...)",
            "made_of":  "what it is made of or consists of",
        },
        "form": {
            "size":  "its typical size",
            "shape": "its typical shape",
            "color": "its typical color",
        },
    },
    "quality": {
        "value": {
            "goodness": "good or bad, or a word for how it is judged",
            "strength": "how strong or intense it is",
        },
    },
    "mind": {
        "inner": {
            "feeling": "the feeling it carries or causes",
            "thought": "the thought or idea it involves",
            "want":    "what it wants, or what is wanted",
        },
    },
    "action": {
        "event": {
            "does":     "what it does",
            "done_to":  "what is done to it",
            "used_for": "what it is used for or good for",
        },
    },
    "relation": {
        "links": {
            "opposite": "its opposite",
            "similar":  "the closest word in meaning",
            "part_of":  "the larger thing it is part of",
            "has_part": "a main part it has",
            "with":     "what it is most often found or used with",
        },
    },
    "context": {
        "setting": {
            "place": "where it is or happens",
            "time":  "when it is or happens",
            "cause": "what causes it, or what it causes",
        },
    },
}

ROLE_PATHS = [(l, s, r) for l, segs in SCHEMA.items()
              for s, roles in segs.items() for r in roles]
assert len(ROLE_PATHS) == 21, len(ROLE_PATHS)
PATHS = [f"{l}.{s}.{r}" for l, s, r in ROLE_PATHS]

VALUE_RE = re.compile(r"^[a-z]+$")


def validate_glyph(word: str, glyph: dict) -> list[str]:
    """Check one glyph; return a list of problems (empty = valid)."""
    problems = []
    for path in PATHS:
        e = glyph.get(path)
        if e is None:
            problems.append(f"{path}: missing")
            continue
        v, w = e["value"], e["weight"]
        if not 0.0 <= w <= 1.0:
            problems.append(f"{path}: weight {w} out of range")
        if v == "none":
            if w != 0.0:
                problems.append(f"{path}: none must have weight 0")
        elif not VALUE_RE.match(v):
            problems.append(f"{path}: value {v!r} is not one lowercase word")
        elif v == word:
            problems.append(f"{path}: a word may not describe itself")
    if not any(glyph[p]["value"] != "none" for p in PATHS if p in glyph):
        problems.append("glyph is empty")
    return problems
