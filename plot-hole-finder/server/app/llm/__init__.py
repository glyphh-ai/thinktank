"""LLM extraction & verification. Claude is the only implementation -- there is
no non-LLM fallback; extraction quality is the product."""

from ..config import Settings, get_settings
from .base import (
    ChunkExtraction,
    ExtractedEntity,
    ExtractedFact,
    Extractor,
    FixSuggestion,
    ScanResult,
    Verifier,
    VerifierVerdict,
    WorldModelInconsistency,
)


def _require_key(settings: Settings) -> None:
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required -- there is no extraction fallback")


def get_extractor(settings: Settings | None = None) -> Extractor:
    settings = settings or get_settings()
    _require_key(settings)
    from .claude import ClaudeExtractor

    return ClaudeExtractor(settings)


def get_verifier(settings: Settings | None = None) -> Verifier:
    settings = settings or get_settings()
    _require_key(settings)
    from .claude import ClaudeVerifier

    return ClaudeVerifier(settings)


__all__ = [
    "Extractor",
    "Verifier",
    "ExtractedEntity",
    "ExtractedFact",
    "ChunkExtraction",
    "VerifierVerdict",
    "FixSuggestion",
    "ScanResult",
    "WorldModelInconsistency",
    "get_extractor",
    "get_verifier",
]
