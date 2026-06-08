"""The extracted world model and analysis results.

This is the structured store from the design: entities, their attribute/event
facts, the analysis runs that process them, and the issues (plot holes) found --
each with evidence spans that point back into the source text.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..plans import PlanTier
from .base import Base, PrimaryKey, Timestamps


class EntityType(str, enum.Enum):
    CHARACTER = "character"
    OBJECT = "object"
    LOCATION = "location"
    CONCEPT = "concept"


class Entity(Base, PrimaryKey, Timestamps):
    __tablename__ = "entities"

    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[EntityType] = mapped_column(Enum(EntityType, native_enum=False, length=16), nullable=False)
    aliases: Mapped[list] = mapped_column(JSON, default=list)  # coref surface forms

    facts: Mapped[list["Fact"]] = relationship(back_populates="entity", cascade="all, delete-orphan")


class FactKind(str, enum.Enum):
    ATTRIBUTE = "attribute"   # entity | key | value  (e.g. eye_color = brown)
    EVENT = "event"           # action involving entity (incl. transformations)
    RELATION = "relation"     # entity | rel | other-entity
    REFERENCE = "reference"   # setup / payoff mention
    KNOWLEDGE = "knowledge"   # entity knows fact F as of time T


class Fact(Base, PrimaryKey, Timestamps):
    """A single extracted assertion, traceable to a source span."""

    __tablename__ = "facts"

    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    entity_id: Mapped[str | None] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"))

    kind: Mapped[FactKind] = mapped_column(Enum(FactKind, native_enum=False, length=16), nullable=False)
    key: Mapped[str] = mapped_column(String(128), nullable=False)     # role / attribute / relation
    value: Mapped[str] = mapped_column(String(512), nullable=False)   # filler
    polarity: Mapped[bool] = mapped_column(default=True, nullable=False)  # False == negated

    narrative_time: Mapped[int] = mapped_column(Integer, default=0)   # order in the text
    story_time: Mapped[int | None] = mapped_column(Integer)           # in-world time (flashbacks)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    entity: Mapped[Entity | None] = relationship(back_populates="facts")


class RunKind(str, enum.Enum):
    FIND = "find"   # detect issues
    FIX = "fix"     # generate fixes (Pro)


class RunStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AnalysisRun(Base, PrimaryKey, Timestamps):
    __tablename__ = "analysis_runs"

    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    kind: Mapped[RunKind] = mapped_column(Enum(RunKind, native_enum=False, length=8), nullable=False)
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, native_enum=False, length=12), default=RunStatus.QUEUED, nullable=False
    )
    plan_at_run: Mapped[PlanTier] = mapped_column(Enum(PlanTier, native_enum=False, length=16))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stats: Mapped[dict] = mapped_column(JSON, default=dict)  # counts, timings, token usage
    error: Mapped[str | None] = mapped_column(Text)

    issues: Mapped[list["Issue"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class IssueCategory(str, enum.Enum):
    CONTINUITY = "continuity"
    BROKEN_REFERENCE = "broken_reference"
    TIMELINE = "timeline"
    CHARACTER = "character"
    KNOWLEDGE = "knowledge"


class Severity(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IssueStatus(str, enum.Enum):
    OPEN = "open"
    DISMISSED = "dismissed"
    FIXED = "fixed"


class Issue(Base, PrimaryKey, Timestamps):
    __tablename__ = "issues"

    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    category: Mapped[IssueCategory] = mapped_column(Enum(IssueCategory, native_enum=False, length=20))
    severity: Mapped[Severity] = mapped_column(Enum(Severity, native_enum=False, length=8))
    status: Mapped[IssueStatus] = mapped_column(
        Enum(IssueStatus, native_enum=False, length=12), default=IssueStatus.OPEN, nullable=False
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    suggested_fix: Mapped[str | None] = mapped_column(Text)  # populated by FIX runs (Pro)

    run: Mapped[AnalysisRun] = relationship(back_populates="issues")
    evidence: Mapped[list["IssueEvidence"]] = relationship(
        back_populates="issue", cascade="all, delete-orphan"
    )


class EvidenceRole(str, enum.Enum):
    STATE_A = "state_a"   # the two sides of a contradiction
    STATE_B = "state_b"
    SETUP = "setup"
    PAYOFF = "payoff"
    CONTEXT = "context"


class IssueEvidence(Base, PrimaryKey, Timestamps):
    """A source span supporting an issue -- enables jump-to-passage in the UI."""

    __tablename__ = "issue_evidence"

    issue_id: Mapped[str] = mapped_column(ForeignKey("issues.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"))
    role: Mapped[EvidenceRole] = mapped_column(Enum(EvidenceRole, native_enum=False, length=12))
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    quote: Mapped[str] = mapped_column(Text, nullable=False)

    issue: Mapped[Issue] = relationship(back_populates="evidence")
