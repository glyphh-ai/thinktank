"""Deterministic test doubles for pipeline *plumbing* tests.

These are NOT product backends and live only in the test tree. They let us
exercise the cortex / persistence / jobs / API end to end without calling Claude
or downloading an embedding model. Detection *quality* is measured against real
Claude in the eval harness, never against these stubs.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from app.embeddings.base import Embedder, l2_normalize
from app.llm.base import (
    ChunkExtraction,
    ExtractedEntity,
    ExtractedFact,
    Extractor,
    FixSuggestion,
    Verifier,
    VerifierVerdict,
    WorldModelInconsistency,
)

# Minimal attribute/event patterns -- just enough to drive the plumbing tests.
_ATTR = [
    (re.compile(r"(\b[A-Z]\w+)'s eyes were (\w+)", re.I), "EYE_COLOR", 1, 2),
    (re.compile(r"(\b[A-Z]\w+) had (\w+) eyes", re.I), "EYE_COLOR", 1, 2),
    (re.compile(r"(\b[A-Z]\w+)'s hair was (\w+)", re.I), "HAIR", 1, 2),
    (re.compile(r"(\b[A-Z]\w+) had (\w+) hair", re.I), "HAIR", 1, 2),
    (re.compile(r"(\b[A-Z]\w+) was (\d+) years old", re.I), "AGE", 1, 2),
]
_EVENT = [
    (re.compile(r"(\b[A-Z]\w+) dyed (?:his|her|their) hair (\w+)", re.I), "DYE_HAIR", 1, 2),
]


class StubExtractor(Extractor):
    def extract(self, chunk_text: str, char_offset: int) -> ChunkExtraction:
        entities: dict[str, ExtractedEntity] = {}
        facts: list[ExtractedFact] = []
        for pattern, key, ng, vg in _ATTR + _EVENT:
            kind = "event" if key in {"DYE_HAIR"} else "attribute"
            for m in pattern.finditer(chunk_text):
                name = m.group(ng).capitalize()
                entities.setdefault(name, ExtractedEntity(name=name, type="character"))
                facts.append(ExtractedFact(
                    entity=name, kind=kind, key=key, value=m.group(vg).upper(),
                    char_start=char_offset + m.start(), char_end=char_offset + m.end(),
                ))
        return ChunkExtraction(entities=list(entities.values()), facts=facts)


class StubVerifier(Verifier):
    def verify(self, candidate: dict, evidence: list[str]) -> VerifierVerdict:
        a, b = candidate.get("value_a", ""), candidate.get("value_b", "")
        confirmed = bool(a) and bool(b) and a != b
        return VerifierVerdict(
            confirmed=confirmed, severity="high",
            title=f"{candidate.get('entity')}'s {candidate.get('key', '').lower()} "
                  f"changes ({a} -> {b})",
            explanation=f"{candidate.get('entity')} {candidate.get('key')}: {a} vs {b}.",
            confidence=0.9 if confirmed else 0.0,
        )

    def suggest_fix(self, issue: dict, evidence: list[str]) -> FixSuggestion:
        return FixSuggestion(text=f"Make {issue.get('entity')} consistent.")

    def review_world_model(self, world_model: str) -> list[WorldModelInconsistency]:
        return []  # holistic arm is a no-op under the stub; quality is Claude-only


class StubEmbedder(Embedder):
    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"\w+", text.lower()):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
                out[i, h % self.dim] += 1.0
        return l2_normalize(out)
