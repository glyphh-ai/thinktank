"""Text utilities: word counting and structural chunking.

Chunking is paragraph-aware (splits on blank lines, packs into ~target-word
chunks) so each chunk keeps narrative coherence and a char offset back into the
source -- which the evidence/jump-to-passage feature relies on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"\b\w+\b")
_PARA_RE = re.compile(r"\n\s*\n")


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


@dataclass
class Chunk:
    index: int
    char_start: int
    char_end: int
    text: str


def chunk_text(text: str, target_words: int = 400) -> list[Chunk]:
    """Split text into ~``target_words`` chunks on paragraph boundaries."""
    chunks: list[Chunk] = []
    pos = 0
    buf_start = 0
    buf_parts: list[str] = []
    buf_words = 0

    def flush(end: int) -> None:
        nonlocal buf_parts, buf_words, buf_start
        if not buf_parts:
            return
        body = "".join(buf_parts)
        chunks.append(Chunk(len(chunks), buf_start, end, body.strip()))
        buf_parts = []
        buf_words = 0

    # Iterate paragraphs while tracking absolute offsets.
    for m in _PARA_RE.finditer(text):
        para = text[pos:m.start()]
        if not buf_parts:
            buf_start = pos
        buf_parts.append(text[pos:m.end()])
        buf_words += word_count(para)
        pos = m.end()
        if buf_words >= target_words:
            flush(m.start())
            buf_start = pos

    # Trailing remainder.
    if pos < len(text):
        if not buf_parts:
            buf_start = pos
        buf_parts.append(text[pos:])
    flush(len(text))
    return [c for c in chunks if c.text]
