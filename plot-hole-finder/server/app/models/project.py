"""Projects and their source documents.

A *project* is one unit of analysis -- an uploaded manuscript or pasted text.
Raw bytes live in blob storage (R2 in prod, local disk in dev); the DB holds the
metadata and the blob key.
"""

from __future__ import annotations

import enum

from sqlalchemy import Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, PrimaryKey, Timestamps


class ProjectStatus(str, enum.Enum):
    CREATED = "created"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    READY = "ready"
    FAILED = "failed"


class SourceType(str, enum.Enum):
    UPLOAD = "upload"
    PASTE = "paste"


class Project(Base, PrimaryKey, Timestamps):
    __tablename__ = "projects"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[ProjectStatus] = mapped_column(
        Enum(ProjectStatus, native_enum=False, length=16), default=ProjectStatus.CREATED, nullable=False
    )
    word_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)

    documents: Mapped[list["Document"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class Document(Base, PrimaryKey, Timestamps):
    __tablename__ = "documents"

    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), default="text/plain", nullable=False)
    source_type: Mapped[SourceType] = mapped_column(
        Enum(SourceType, native_enum=False, length=16), nullable=False
    )
    blob_key: Mapped[str] = mapped_column(String(1024), nullable=False)  # original bytes
    # Extracted plain text (PDFs/Word are parsed at upload); paste reuses blob_key.
    text_blob_key: Mapped[str | None] = mapped_column(String(1024))
    byte_size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)  # dedupe / integrity

    project: Mapped[Project] = relationship(back_populates="documents")
