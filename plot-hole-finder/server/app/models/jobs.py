"""Durable background jobs -- the source of truth for async work.

Robustness comes from this table, not the dispatch mechanism: status, attempts,
lease (for heartbeat / dead-worker reclaim), backoff schedule, progress, and
result all persist, so jobs survive restarts, retry on failure, and are
observable regardless of how workers are deployed (process, container, K8s pod).
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, PrimaryKey, Timestamps


class JobType(str, enum.Enum):
    ANALYZE = "analyze"   # run the find pipeline
    FIX = "fix"           # Pro: generate suggested fixes


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class Job(Base, PrimaryKey, Timestamps):
    __tablename__ = "jobs"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    type: Mapped[JobType] = mapped_column(Enum(JobType, native_enum=False, length=16), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False, length=12), default=JobStatus.QUEUED, nullable=False, index=True
    )
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)

    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    step: Mapped[str | None] = mapped_column(String(64))

    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    # Eligible to run when status==queued and scheduled_at<=now (enables backoff).
    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    # While running, the worker holds a lease; if it lapses the job is reclaimed.
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # The AnalysisRun this job produced (for linking results).
    run_id: Mapped[str | None] = mapped_column(ForeignKey("analysis_runs.id", ondelete="SET NULL"))
