"""The scalable memory backend for rolling detection.

Instead of re-sending the whole running memory on every chunk (O(n^2), and it
eventually exceeds the context window), facts are persisted here and *retrieved*
per chunk by entity. Per-step cost stays bounded, so detection scales to
documents far larger than any context window.
"""

from __future__ import annotations

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, PrimaryKey, Timestamps


class MemoryFact(Base, PrimaryKey, Timestamps):
    __tablename__ = "memory_facts"

    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    entity: Mapped[str] = mapped_column(String(255), index=True)  # surface name, retrieval key
    key: Mapped[str] = mapped_column(String(128))                 # attribute/relation
    value: Mapped[str] = mapped_column(String(1024))
    kind: Mapped[str] = mapped_column(String(16), default="state")  # state | event
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
