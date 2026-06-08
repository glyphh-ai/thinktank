"""Claude-backed Extractor/Verifier (Anthropic API) for production.

Lazy-imports the ``anthropic`` SDK so it's only required when selected. Prompts
ask for strict JSON, which we parse into the same dataclasses the heuristic
implementations produce, so the pipeline is provider-agnostic.
"""

from __future__ import annotations

import json

from ..config import Settings
from .base import (
    ChunkExtraction,
    ExtractedEntity,
    ExtractedFact,
    Extractor,
    FixSuggestion,
    Verifier,
    VerifierVerdict,
)

_EXTRACT_SYSTEM = (
    "You extract a compact world-model from a passage of narrative or legal text "
    "for consistency checking. Return ONLY JSON: "
    "{\"entities\":[{\"name\",\"type\"}],\"facts\":[{\"entity\",\"kind\",\"key\","
    "\"value\"}]}. type in [character,object,location,concept]; kind in "
    "[attribute,event,relation,knowledge]. "
    "Extract ONLY facts that could later contradict: stable attributes (eye/hair "
    "colour, age, height, title or role, possessions), locations, relationships, "
    "knowledge a character holds, and transformation events that change an "
    "attribute (dyeing hair, a death, an object destroyed). Use a stable "
    "lowercase snake_case key (e.g. eye_color, age, location, role). Do NOT "
    "extract atmosphere, mood, routine actions, or one-off descriptions. Emit one "
    "fact per distinct attribute assertion; keep values short (1-3 words)."
)

_VERIFY_SYSTEM = (
    "You are a fact-checker for narrative continuity. Given a candidate "
    "inconsistency and evidence passages, decide if it is a genuine plot hole "
    "(no in-story event explains it). Return ONLY JSON: {\"confirmed\":bool,"
    "\"severity\":\"low|medium|high\",\"title\":str,\"explanation\":str,"
    "\"confidence\":number}."
)


class _ClaudeBase:
    def __init__(self, settings: Settings):
        import anthropic  # lazy

        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_model
        self._extract_max_tokens = getattr(settings, "extract_max_tokens", 8000)
        # Cumulative token accounting (for cost tracking / per-run stats).
        self.usage = {
            "calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        }

    def _json(self, system: str, user: str, max_tokens: int = 4096) -> dict:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        u = msg.usage
        self.usage["calls"] += 1
        self.usage["input_tokens"] += u.input_tokens
        self.usage["output_tokens"] += u.output_tokens
        self.usage["cache_read_input_tokens"] += getattr(u, "cache_read_input_tokens", 0) or 0
        self.usage["cache_creation_input_tokens"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        text = "".join(block.text for block in msg.content if block.type == "text")
        return json.loads(_strip_fences(text))


class ClaudeExtractor(_ClaudeBase, Extractor):
    def extract(self, chunk_text: str, char_offset: int) -> ChunkExtraction:
        # Offsets are recomputed locally by the pipeline (the LLM is unreliable at
        # exact offsets and they're the bulk of output cost), so we don't ask for
        # them here -- char_start/char_end stay 0 and signal "locate me".
        data = self._json(_EXTRACT_SYSTEM, chunk_text, max_tokens=self._extract_max_tokens)
        entities = [
            ExtractedEntity(name=e["name"], type=e.get("type", "character"),
                            aliases=e.get("aliases", []))
            for e in data.get("entities", [])
        ]
        facts = [
            ExtractedFact(entity=f["entity"], kind=f.get("kind", "attribute"),
                          key=f["key"], value=str(f["value"]))
            for f in data.get("facts", [])
        ]
        return ChunkExtraction(entities=entities, facts=facts)


class ClaudeVerifier(_ClaudeBase, Verifier):
    def verify(self, candidate: dict, evidence: list[str]) -> VerifierVerdict:
        user = json.dumps({"candidate": candidate, "evidence": evidence})
        d = self._json(_VERIFY_SYSTEM, user, max_tokens=1024)
        return VerifierVerdict(
            confirmed=bool(d["confirmed"]), severity=d.get("severity", "medium"),
            title=d.get("title", ""), explanation=d.get("explanation", ""),
            confidence=float(d.get("confidence", 0.0)),
        )

    def suggest_fix(self, issue: dict, evidence: list[str]) -> FixSuggestion:
        user = json.dumps({"issue": issue, "evidence": evidence})
        d = self._json(
            "Propose a concise edit resolving this narrative inconsistency. "
            "Return ONLY JSON: {\"text\":str}.",
            user, max_tokens=1024,
        )
        return FixSuggestion(text=d.get("text", ""))


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return text.strip()
