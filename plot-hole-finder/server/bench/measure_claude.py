"""Measure Claude extraction token usage and cost on a sample document.

Usage (needs LLM_BACKEND=claude + ANTHROPIC_API_KEY in env/.env):

    python -m bench.measure_claude            # built-in synthetic ~3k-word doc
    python -m bench.measure_claude path.txt   # measure a real document

Reports per-chunk extraction tokens, total cost at the configured model's
pricing, and a linear projection to a 200k-word manuscript -- plus the cost of
sending the whole document to Claude in a single long-context pass, for
comparison. Numbers are empirical, not estimates.
"""

from __future__ import annotations

import sys
import time

from app.config import get_settings
from app.llm.claude import ClaudeExtractor
from app.text import chunk_text, word_count

# Per-1M-token pricing by model (input, output). Source: claude-api skill.
PRICING = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def _sample_document() -> str:
    chars = ["Elena", "Marcus", "Dr. Vance", "the Captain", "Iris"]
    base = (
        "{c} walked into the hall, eyes scanning the crowd. The room smelled of old "
        "paper and rain. {c} had travelled far, and the journey showed in the dust on "
        "her coat. A fire crackled in the hearth, throwing long shadows across the stone "
        "floor. Conversations rose and fell like tide. "
    )
    scenes = [base.format(c=chars[i % len(chars)]) + f"This was gathering {i + 1}."
              for i in range(40)]
    scenes.insert(2, "Elena's eyes were green, bright as spring, when she arrived at the keep.")
    scenes.append("In the final chapter, Elena's eyes were blue, cold as winter, as she left.")
    return "\n\n".join(scenes)


def main() -> None:
    settings = get_settings()
    in_per_m, out_per_m = PRICING.get(settings.anthropic_model, (3.00, 15.00))

    doc = open(sys.argv[1]).read() if len(sys.argv) > 1 else _sample_document()
    wc = word_count(doc)
    chunks = chunk_text(doc)
    print(f"model: {settings.anthropic_model} | {wc:,} words | {len(chunks)} chunks")

    ext = ClaudeExtractor(settings)
    whole = ext._client.messages.count_tokens(
        model=settings.anthropic_model, messages=[{"role": "user", "content": doc}]
    ).input_tokens

    t0 = time.time()
    entities = facts = 0
    for c in chunks:
        r = ext.extract(c.text, c.char_start)
        entities += len(r.entities)
        facts += len(r.facts)
    elapsed = time.time() - t0

    u = ext.usage
    cost = u["input_tokens"] / 1e6 * in_per_m + u["output_tokens"] / 1e6 * out_per_m
    scale = 200_000 / wc

    print(f"\nextraction: {u['calls']} calls | in {u['input_tokens']:,} / out "
          f"{u['output_tokens']:,} tok | {entities} entities, {facts} facts | {elapsed:.0f}s")
    print(f"cost: ${cost:.4f}  (${cost / wc * 1000:.4f} per 1k words)")
    print(f"\nprojected to 200k words:")
    print(f"  chunked extraction : ~${cost * scale:.2f}  (~{u['calls'] * scale:.0f} calls)")
    print(f"  single long-context pass (input only): ~${whole * scale / 1e6 * in_per_m:.2f}")


if __name__ == "__main__":
    main()
