"""LLM extraction & verification, provider-agnostic."""

from ..config import Settings, get_settings
from .base import (
    ChunkExtraction,
    ExtractedEntity,
    ExtractedFact,
    Extractor,
    FixSuggestion,
    Verifier,
    VerifierVerdict,
)
from .heuristic import HeuristicExtractor, HeuristicVerifier


def get_extractor(settings: Settings | None = None) -> Extractor:
    settings = settings or get_settings()
    if settings.llm_backend == "claude":
        from .claude import ClaudeExtractor

        return ClaudeExtractor(settings)
    return HeuristicExtractor()


def get_verifier(settings: Settings | None = None) -> Verifier:
    settings = settings or get_settings()
    if settings.llm_backend == "claude":
        from .claude import ClaudeVerifier

        return ClaudeVerifier(settings)
    return HeuristicVerifier()


__all__ = [
    "Extractor",
    "Verifier",
    "ExtractedEntity",
    "ExtractedFact",
    "ChunkExtraction",
    "VerifierVerdict",
    "FixSuggestion",
    "get_extractor",
    "get_verifier",
]
