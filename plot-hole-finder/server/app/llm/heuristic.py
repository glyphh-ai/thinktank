"""Offline, deterministic Extractor/Verifier for dev and tests.

The extractor recognises a handful of attribute and transformation-event
patterns with regexes -- enough to drive the full pipeline end to end (and to
plant/detect contradictions in tests) without any LLM. The verifier confirms a
candidate when the two evidence sides genuinely disagree.
"""

from __future__ import annotations

import re

from .base import (
    ChunkExtraction,
    ExtractedEntity,
    ExtractedFact,
    Extractor,
    FixSuggestion,
    Verifier,
    VerifierVerdict,
)

# (pattern, key, name_group, value_group). Names become character entities.
_ATTR_PATTERNS = [
    (re.compile(r"(\b[A-Z]\w+)'s eyes were (\w+)", re.I), "EYE_COLOR", 1, 2),
    (re.compile(r"(\b[A-Z]\w+) had (\w+) eyes", re.I), "EYE_COLOR", 1, 2),
    (re.compile(r"(\b[A-Z]\w+)'s hair was (\w+)", re.I), "HAIR", 1, 2),
    (re.compile(r"(\b[A-Z]\w+) had (\w+) hair", re.I), "HAIR", 1, 2),
    (re.compile(r"(\b[A-Z]\w+) was (\d+) years old", re.I), "AGE", 1, 2),
]

# Transformation events that *license* an attribute change between two states.
_EVENT_PATTERNS = [
    (re.compile(r"(\b[A-Z]\w+) dyed (?:his|her|their) hair (\w+)", re.I), "DYE_HAIR", 1, 2),
    (re.compile(r"(\b[A-Z]\w+) (?:put in|wore) (\w+) contacts", re.I), "WEAR_CONTACTS", 1, 2),
]


class HeuristicExtractor(Extractor):
    def extract(self, chunk_text: str, char_offset: int) -> ChunkExtraction:
        entities: dict[str, ExtractedEntity] = {}
        facts: list[ExtractedFact] = []

        def note_entity(name: str) -> None:
            entities.setdefault(name, ExtractedEntity(name=name, type="character"))

        for pattern, key, ng, vg in _ATTR_PATTERNS:
            for m in pattern.finditer(chunk_text):
                name = m.group(ng).capitalize()
                note_entity(name)
                facts.append(ExtractedFact(
                    entity=name, kind="attribute", key=key, value=m.group(vg).upper(),
                    char_start=char_offset + m.start(), char_end=char_offset + m.end(),
                ))

        for pattern, key, ng, vg in _EVENT_PATTERNS:
            for m in pattern.finditer(chunk_text):
                name = m.group(ng).capitalize()
                note_entity(name)
                facts.append(ExtractedFact(
                    entity=name, kind="event", key=key, value=m.group(vg).upper(),
                    char_start=char_offset + m.start(), char_end=char_offset + m.end(),
                ))

        return ChunkExtraction(entities=list(entities.values()), facts=facts)


class HeuristicVerifier(Verifier):
    def verify(self, candidate: dict, evidence: list[str]) -> VerifierVerdict:
        a = candidate.get("value_a", "")
        b = candidate.get("value_b", "")
        confirmed = bool(a) and bool(b) and a != b
        return VerifierVerdict(
            confirmed=confirmed,
            severity="high" if candidate.get("key") in {"EYE_COLOR", "AGE"} else "medium",
            title=f"{candidate.get('entity', 'Entity')}'s {candidate.get('key', '').lower()} "
                  f"changes ({a} -> {b})",
            explanation=(
                f"{candidate.get('entity')} is described with {candidate.get('key')}={a} and "
                f"later {candidate.get('key')}={b}, with no event explaining the change."
            ),
            confidence=0.9 if confirmed else 0.0,
        )

    def suggest_fix(self, issue: dict, evidence: list[str]) -> FixSuggestion:
        return FixSuggestion(
            text=(
                f"Make {issue.get('entity')}'s {issue.get('key', '').lower()} consistent: "
                f"use a single value throughout, or add an event explaining the change."
            )
        )
