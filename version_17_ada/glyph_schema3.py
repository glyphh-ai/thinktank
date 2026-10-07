"""
THE GLYPH SCHEMA v3 — identical atomic makeup
=============================================
Same layers, segments and role names as v2 (so the same 40 schema words and
the same keys), but every glyph has EXACTLY the same makeup:

  - all 27 roles are filled with a real word — "none" is not allowed
  - no weights: every role weighs 1/27

So every glyph is 27 binds × 1/27, and two words differ only in WHICH words
fill the roles. Every role description is rewritten to apply to every word
(v2's tense/degree/phrase/... did not apply to all words, which forced
"none" and gave love 23 real binds but hate 18).
"""

import re

from glyph_schema2 import SCHEMA as _V2

SCHEMA = {
    "grammar": {
        "kind": {
            "class": "its word class: noun, verb, adjective, adverb, pronoun, preposition, conjunction, determiner, interjection",
            "base":  "its root word (ran -> run, cats -> cat); the word itself if it is already the root",
        },
        "form": {
            "number": "how many it involves or refers to: one, some, many, all",
            "tense":  "the time it points to: past, present, future, always",
            "degree": "how far it goes on its scale: little, some, very, most",
        },
    },
    "sentence": {
        "position": {
            "slot":   "its usual job in a sentence: subject, object, action, modifier, link, connector",
            "before": "the word that most often comes right before it",
            "after":  "the word that most often comes right after it",
        },
        "frame": {
            "agent":   "who or what typically does it, has it, uses it, or is it",
            "patient": "who or what it is typically done to, aimed at, or used on",
            "partner": "the word it most often goes together with",
        },
    },
    "idiom": {
        "expression": {
            "phrase": "the key other word of a common phrase or saying it appears in",
            "figure": "what it means inside that phrase",
        },
        "tone": {
            "register": "its tone: plain, formal, slang, childish, poetic",
            "mood":     "the mood it sets",
        },
    },
    "meaning": {
        "sense": {
            "category": "what kind of thing it is (feeling, animal, tool, action, place, color, word, ...)",
            "feeling":  "the feeling it carries or causes",
            "goodness": "how it is judged: good, bad, or a word for its value",
            "strength": "how strong or intense it is",
        },
        "relation": {
            "opposite": "its opposite, or the thing furthest from it",
            "similar":  "the closest word in meaning",
            "part":     "a main part it has (figurative for abstract things)",
            "whole":    "the larger thing it belongs to",
        },
        "context": {
            "place":   "where it is or happens",
            "time":    "when it is or happens",
            "cause":   "what causes it or brings it about",
            "purpose": "what it is for, or what it leads to",
        },
    },
}

# identical names to v2 — only the descriptions changed
assert {(l, s, r) for l in SCHEMA for s in SCHEMA[l] for r in SCHEMA[l][s]} == \
       {(l, s, r) for l in _V2 for s in _V2[l] for r in _V2[l][s]}

ROLE_PATHS = [(l, s, r) for l, segs in SCHEMA.items()
              for s, roles in segs.items() for r in roles]
assert len(ROLE_PATHS) == 27, len(ROLE_PATHS)
PATHS = [f"{l}.{s}.{r}" for l, s, r in ROLE_PATHS]
SCHEMA_WORDS = sorted({w for path in ROLE_PATHS for w in path})

WEIGHTED = False   # every role weighs the same

PROMPT_INTRO = """You write the glyph of an English word: what it is as a word,
how it is used in sentences, what it means in common phrases, and what it
means — as role: value pairs in a fixed schema that is the same for every
word. Every glyph has exactly the same makeup: EVERY role gets a real word."""

PROMPT_EXAMPLES = """- "none" is NOT allowed. If a role seems not to fit, give the closest
  fitting word — figurative is fine (love -> part: heart, whole: life).
- "ran" -> class: verb, base: run, tense: past. "love" -> base: love.
- Function words fill every role too: "the" -> class: determiner, after: dog,
  category: word, feeling: plain, opposite: a."""

VALUE_RE = re.compile(r"^[a-z]+$")


def validate_glyph(word: str, glyph: dict) -> list[str]:
    problems = []
    for path in PATHS:
        e = glyph.get(path)
        if e is None:
            problems.append(f"{path}: missing")
        elif e["value"] == "none":
            problems.append(f"{path}: none is not allowed")
        elif not VALUE_RE.match(e["value"]):
            problems.append(f"{path}: value {e['value']!r} is not one lowercase word")
    return problems
