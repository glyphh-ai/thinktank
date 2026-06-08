"""Declarative base, shared mixins, and portable column types.

Portability is the whole point of the "bones": the same models run on SQLite in
dev and Postgres (+ pgvector) in prod.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import numpy as np
from sqlalchemy import DateTime, LargeBinary, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


class Base(DeclarativeBase):
    pass


def new_id() -> str:
    return uuid.uuid4().hex


class PrimaryKey:
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)


class Timestamps:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Embedding(TypeDecorator):
    """RAG passage embedding: a real ``vector(dim)`` on Postgres (pgvector),
    transparently stored as float32 bytes on SQLite so dev needs no extension.

    pgvector is enabled *from day one* wherever Postgres is the backend; the
    SQLite fallback keeps local dev infra-free. (HDC hypervectors do NOT use
    this type -- see :class:`Bipolar`.)
    """

    impl = LargeBinary
    cache_ok = True

    def __init__(self, dim: int):
        self.dim = dim
        super().__init__()

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector  # lazy: only needed on PG

            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(LargeBinary())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        if dialect.name == "postgresql":
            return arr  # pgvector accepts ndarray/list directly
        return arr.tobytes()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return np.asarray(value, dtype=np.float32)
        return np.frombuffer(value, dtype=np.float32)


class Bipolar(TypeDecorator):
    """A glyphh HDC hypervector stored as int8 bytes on any backend.

    These are high-dimensional (~10k) bipolar vectors consumed by the in-process
    cortex, not ANN-queried in SQL, so they are intentionally NOT pgvector.
    """

    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return np.asarray(value, dtype=np.int8).tobytes()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return np.frombuffer(value, dtype=np.int8)
