"""Durable job queue operations (DB-backed, broker-free).

Lease-based claiming: a worker flips a queued job to running and takes a short
lease; long work happens outside any row lock, with the lease extended via
heartbeats. If a worker dies, ``reclaim_stale`` requeues its lapsed jobs. Failed
jobs retry with exponential backoff until ``max_attempts``.

On Postgres, claiming uses ``FOR UPDATE SKIP LOCKED`` for safe multi-worker
concurrency; on SQLite the single-writer lock suffices for dev.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Job, JobStatus, JobType


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def enqueue(
    session: Session,
    user_id: str,
    job_type: JobType,
    *,
    project_id: str | None = None,
    payload: dict | None = None,
    max_attempts: int = 3,
) -> Job:
    job = Job(
        user_id=user_id, project_id=project_id, type=job_type,
        status=JobStatus.QUEUED, payload=payload or {}, max_attempts=max_attempts,
        scheduled_at=_now(),
    )
    session.add(job)
    session.flush()
    return job


def claim_next(session: Session, lease_seconds: int, types: list[JobType] | None = None) -> Job | None:
    """Atomically claim the oldest eligible queued job, or return None."""
    now = _now()
    stmt = (
        select(Job)
        .where(Job.status == JobStatus.QUEUED, Job.scheduled_at <= now)
        .order_by(Job.scheduled_at, Job.created_at)
        .limit(1)
    )
    if types:
        stmt = stmt.where(Job.type.in_(types))
    if session.bind.dialect.name == "postgresql":
        stmt = stmt.with_for_update(skip_locked=True)

    job = session.scalar(stmt)
    if job is None:
        return None
    job.status = JobStatus.RUNNING
    job.attempts += 1
    job.lease_expires_at = now + timedelta(seconds=lease_seconds)
    session.flush()
    return job


def heartbeat(session: Session, job: Job, lease_seconds: int, progress: float | None = None,
              step: str | None = None) -> None:
    """Extend the lease and (optionally) record progress; commits immediately."""
    job.lease_expires_at = _now() + timedelta(seconds=lease_seconds)
    if progress is not None:
        job.progress = max(0.0, min(1.0, progress))
    if step is not None:
        job.step = step
    session.commit()


def complete(session: Session, job: Job, result: dict, run_id: str | None = None) -> None:
    job.status = JobStatus.SUCCEEDED
    job.progress = 1.0
    job.result = result
    job.error = None
    job.lease_expires_at = None
    if run_id:
        job.run_id = run_id


def fail(session: Session, job: Job, error: str, backoff_base: int = 5) -> None:
    """Retry with exponential backoff, or mark failed once attempts are spent."""
    job.error = error
    job.lease_expires_at = None
    if job.attempts < job.max_attempts:
        delay = backoff_base * (2 ** (job.attempts - 1))
        job.status = JobStatus.QUEUED
        job.scheduled_at = _now() + timedelta(seconds=delay)
    else:
        job.status = JobStatus.FAILED


def reclaim_stale(session: Session) -> int:
    """Requeue jobs whose worker died (running past their lease)."""
    now = _now()
    stale = session.scalars(
        select(Job).where(Job.status == JobStatus.RUNNING, Job.lease_expires_at.is_not(None))
    ).all()
    count = 0
    for job in stale:
        if _aware(job.lease_expires_at) < now:
            job.status = JobStatus.QUEUED
            job.lease_expires_at = None
            count += 1
    return count


def cancel(session: Session, job: Job) -> None:
    if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        job.status = JobStatus.CANCELED
        job.lease_expires_at = None


def get_job(session: Session, job_id: str) -> Job | None:
    return session.get(Job, job_id)
