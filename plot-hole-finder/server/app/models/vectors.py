"""Vector persistence -- two distinct stores, two column types.

- :class:`PassageEmbedding` -- dense float RAG embeddings, pgvector from day one
  (real ``vector(dim)`` + ANN index on Postgres; bytes fallback on SQLite).
- :class:`HypervectorRecord` -- glyphh HDC cortex artifacts, stored as int8 bytes
  and rehydrated into the in-process engine (NOT pgvector; see base.Bipolar).
"""

from __future__ import annotations

import enum

from sqlalchemy import Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, Bipolar, Embedding, PrimaryKey, Timestamps

# Embedding dimension for RAG retrieval. Matches the small model
# (bge-small-en-v1.5 / all-MiniLM-L6-v2 family = 384). Fixed here because the
# pgvector column dimension is set at DDL time.
EMBEDDING_DIM = 384


class PassageEmbedding(Base, PrimaryKey, Timestamps):
    """A chunk of source text + its embedding, for evidence retrieval (RAG)."""

    __tablename__ = "passage_embeddings"

    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"))
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[object] = mapped_column(Embedding(EMBEDDING_DIM), nullable=False)


class CortexLevel(str, enum.Enum):
    CODEBOOK = "codebook"
    SEGMENT = "segment"
    LAYER = "layer"
    CORTEX = "cortex"


class HypervectorRecord(Base, PrimaryKey, Timestamps):
    """A persisted glyphh node so a project's cortex can be rebuilt from the DB."""

    __tablename__ = "hypervectors"

    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    level: Mapped[CortexLevel] = mapped_column(Enum(CortexLevel, native_enum=False, length=12))
    label: Mapped[str] = mapped_column(String(255), nullable=False)  # node name / symbol
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    data: Mapped[object] = mapped_column(Bipolar, nullable=False)  # int8 bytes
