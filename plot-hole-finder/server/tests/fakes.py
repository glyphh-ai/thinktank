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


_STUB_LICENSING = {"HAIR": {"DYE_HAIR"}, "EYE_COLOR": {"WEAR_CONTACTS"}}


class StubVerifier(Verifier):
    def scan_chunk(self, chunk_text: str, known_facts: list[dict]) -> "ScanResult":
        from app.llm.base import ScanResult

        ex = StubExtractor()
        facts = ex.extract(chunk_text, 0).facts
        state = {(f["entity"], f["key"]): f["value"]
                 for f in known_facts if f.get("kind") != "event"}
        events: dict[str, set] = {}
        for f in known_facts:
            if f.get("kind") == "event":
                events.setdefault(f["entity"], set()).add(f["key"])
        for f in facts:
            if f.kind == "event":
                events.setdefault(f.entity, set()).add(f.key)

        new_facts, issues = [], []
        for f in facts:
            kind = "event" if f.kind == "event" else "state"
            new_facts.append({"entity": f.entity, "key": f.key, "value": f.value, "kind": kind})
            if f.kind != "attribute":
                continue
            prev = state.get((f.entity, f.key))
            if prev is not None and prev != f.value and not (
                    events.get(f.entity, set()) & _STUB_LICENSING.get(f.key, set())):
                issues.append(WorldModelInconsistency(
                    entity=f.entity, key=f.key, value_a=prev, value_b=f.value,
                    category="continuity", severity="high",
                    explanation=f"{f.entity} {f.key}: {prev} -> {f.value}"))
            state[(f.entity, f.key)] = f.value
        return ScanResult(new_facts=new_facts, issues=issues)

    def rolling_scan(self, chunks: list[str]) -> list[WorldModelInconsistency]:
        """Minimal cross-chunk attribute-contradiction detection (test double for
        the real Claude rolling memory), with event-licensing."""
        ex = StubExtractor()
        last: dict[tuple, str] = {}
        events: dict[str, set] = {}
        out = []
        for ch in chunks:
            facts = ex.extract(ch, 0).facts
            for f in facts:
                if f.kind == "event":
                    events.setdefault(f.entity, set()).add(f.key)
            for f in facts:
                if f.kind != "attribute":
                    continue
                prev = last.get((f.entity, f.key))
                if prev is not None and prev != f.value and not (
                        events.get(f.entity, set()) & _STUB_LICENSING.get(f.key, set())):
                    out.append(WorldModelInconsistency(
                        entity=f.entity, key=f.key, value_a=prev, value_b=f.value,
                        category="continuity", severity="high",
                        explanation=f"{f.entity} {f.key}: {prev} -> {f.value}"))
                last[(f.entity, f.key)] = f.value
        return out

    def single_pass(self, document: str) -> list[WorldModelInconsistency]:
        return []  # eval-only arm; no-op under the stub

    def review_world_model(self, world_model: str) -> list[WorldModelInconsistency]:
        return []

    def filter_candidates(self, candidates: list[dict], world_model: str) -> list[dict]:
        # Pass through: structured candidates already had licensing pre-filtered,
        # so confirm those with two distinct values.
        return [c for c in candidates if c.get("value_a") != c.get("value_b")]

    def suggest_fix(self, issue: dict, evidence: list[str]) -> FixSuggestion:
        return FixSuggestion(text=f"Make {issue.get('entity')} consistent.")


class StubEmbedder(Embedder):
    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"\w+", text.lower()):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
                out[i, h % self.dim] += 1.0
        return l2_normalize(out)
