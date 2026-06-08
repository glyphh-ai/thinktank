"""LLM interfaces for the analysis pipeline.

Two roles:
- :class:`Extractor` -- map step: prose chunk -> structured facts (entities,
  attribute/event/reference assertions). Parallel, bounded per chunk.
- :class:`Verifier` -- precision gate: given a candidate inconsistency and its
  evidence passages, confirm/reject, rate severity, and explain.

Prod uses Claude; dev uses deterministic offline implementations so the whole
pipeline runs without an API key.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ExtractedEntity:
    name: str
    type: str  # character | object | location | concept
    aliases: list[str] = field(default_factory=list)


@dataclass
class ExtractedFact:
    entity: str          # canonical/surface name of the subject entity
    kind: str            # attribute | event | relation | reference | knowledge
    key: str             # role / attribute / relation / reference-type
    value: str           # filler
    char_start: int = 0
    char_end: int = 0
    confidence: float = 1.0
    polarity: bool = True


@dataclass
class ChunkExtraction:
    entities: list[ExtractedEntity]
    facts: list[ExtractedFact]


@dataclass
class VerifierVerdict:
    confirmed: bool
    severity: str        # low | medium | high
    title: str
    explanation: str
    confidence: float


@dataclass
class FixSuggestion:
    text: str


@dataclass
class WorldModelInconsistency:
    """An inconsistency found by the holistic pass over the compressed world model."""

    entity: str
    key: str
    value_a: str
    value_b: str
    category: str        # continuity | timeline | character | knowledge | broken_reference
    severity: str        # low | medium | high
    explanation: str
    confidence: float = 0.8


class Extractor(ABC):
    @abstractmethod
    def extract(self, chunk_text: str, char_offset: int) -> ChunkExtraction:
        """Extract entities and facts from one chunk; offsets are absolute."""


class Verifier(ABC):
    @abstractmethod
    def rolling_scan(self, chunks: list[str]) -> list[WorldModelInconsistency]:
        """The detection engine: read chunks in order, maintain a running memory
        of durable facts/constraints, and flag a chunk when it contradicts memory
        with no event explaining it. Scales past the context window."""

    @abstractmethod
    def single_pass(self, document: str) -> list[WorldModelInconsistency]:
        """Recall arm / baseline: read raw text and surface candidates (eval)."""

    @abstractmethod
    def review_world_model(self, world_model: str) -> list[WorldModelInconsistency]:
        """Reason over a compressed world model (eval comparison arm)."""

    @abstractmethod
    def filter_candidates(self, candidates: list[dict], world_model: str) -> list[dict]:
        """Precision filter over candidate inconsistencies (eval comparison arm)."""

    @abstractmethod
    def suggest_fix(self, issue: dict, evidence: list[str]) -> FixSuggestion:
        """Propose an edit that resolves the issue (Pro 'fix this for me')."""
