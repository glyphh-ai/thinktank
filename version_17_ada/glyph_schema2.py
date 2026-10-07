"""
THE GLYPH SCHEMA v2 — grammar → sentence → idiom → meaning
==========================================================
One fixed lattice for every word, four layers from form to sense:

  grammar   what the word IS as a word        (class, base, number, tense, degree)
  sentence  how it is USED in a sentence      (slot, neighbours, agent/patient)
  idiom     what it means in FIXED phrases    (phrase, figure, register, mood)
  meaning   what it MEANS                     (sense, relations, context)

Every layer, segment and role name is ONE lowercase word, and every one of
those words is seeded into the closure (SCHEMA_WORDS) — so the keys of the
key/value binds are glyphs in the same space, not arbitrary labels.

Each role takes one word or "none" and a weight 0..1.
"""

import re

# layer -> {segment -> {role -> what the role asks}}
SCHEMA = {
    "grammar": {
        "kind": {
            "class": "its word class: noun, verb, adjective, adverb, pronoun, preposition, conjunction, determiner, interjection",
            "base":  "its base form if this is an inflected form (ran -> run, cats -> cat); none if it is already the base",
        },
        "form": {
            "number": "singular or plural, if it has number",
            "tense":  "past or present, if it is a verb form",
            "degree": "comparative or superlative, if it is a graded form",
        },
    },
    "sentence": {
        "position": {
            "slot":   "its usual job in a sentence: subject, object, action, modifier, link, connector",
            "before": "the word that most often comes right before it",
            "after":  "the word that most often comes right after it",
        },
        "frame": {
            "agent":   "who or what typically does it, has it, or is it",
            "patient": "who or what it is typically done to or aimed at",
            "partner": "the word it most often goes together with",
        },
    },
    "idiom": {
        "expression": {
            "phrase": "the key other word of its most common fixed expression (love -> fall, as in 'fall in love')",
            "figure": "what it means inside that fixed expression",
        },
        "tone": {
            "register": "its tone: plain, formal, slang, childish, poetic",
            "mood":     "the mood it sets",
        },
    },
    "meaning": {
        "sense": {
            "category": "what kind of thing it is (feeling, animal, tool, action, place, color, ...)",
            "feeling":  "the feeling it carries or causes",
            "goodness": "good or bad, or a word for how it is judged",
            "strength": "how strong or intense it is",
        },
        "relation": {
            "opposite": "its opposite",
            "similar":  "the closest word in meaning",
            "part":     "a main part it has",
            "whole":    "the larger thing it is part of",
        },
        "context": {
            "place":   "where it is or happens",
            "time":    "when it is or happens",
            "cause":   "what causes it",
            "purpose": "what it is for, or what it leads to",
        },
    },
}

ROLE_PATHS = [(l, s, r) for l, segs in SCHEMA.items()
              for s, roles in segs.items() for r in roles]
assert len(ROLE_PATHS) == 27, len(ROLE_PATHS)
PATHS = [f"{l}.{s}.{r}" for l, s, r in ROLE_PATHS]

# Every word the schema is built from — each must have its own glyph.
SCHEMA_WORDS = sorted({w for path in ROLE_PATHS for w in path})

PROMPT_INTRO = """You write the glyph of an English word: what it is as a word,
how it is used in sentences, what it means in fixed expressions, and what it
means — as role: value pairs in a fixed schema that is the same for every
word. Fill EVERY role."""

PROMPT_EXAMPLES = """- Grammar roles describe THIS form of the word: "ran" -> class: verb,
  base: run, tense: past. "love" -> class: noun (its dominant use), base: none.
- Function words get glyphs too: "the" -> class: determiner, slot: modifier,
  after: dog (a typical next word), category: word."""

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
        if v != "none" and not VALUE_RE.match(v):
            problems.append(f"{path}: value {v!r} is not one lowercase word")
    if not any(glyph[p]["value"] != "none" for p in PATHS if p in glyph):
        problems.append("glyph is empty")
    return problems
