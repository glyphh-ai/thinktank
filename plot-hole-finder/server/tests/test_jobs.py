"""Unit tests for the durable job queue service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.config import Settings
from app.db import init_db, make_engine, make_session_factory, session_scope
from app.models import Job, JobStatus, JobType, User
from app.services import jobs as job_service


@pytest.fixture
def factory(tmp_path):
    settings = Settings(database_url=f"sqlite+pysqlite:///{tmp_path}/jobs.db")
    engine = make_engine(settings)
    init_db(engine)
    return make_session_factory(engine)


def _user(session) -> User:
    u = User(email="j@q.com", password_hash="h")
    session.add(u)
    session.flush()
    return u


def test_enqueue_and_claim(factory):
    with session_scope(factory) as s:
        user = _user(s)
        job = job_service.enqueue(s, user.id, JobType.ANALYZE, project_id=None)
        assert job.status is JobStatus.QUEUED

    with session_scope(factory) as s:
        claimed = job_service.claim_next(s, lease_seconds=600)
        assert claimed is not None
        assert claimed.status is JobStatus.RUNNING
        assert claimed.attempts == 1
        assert claimed.lease_expires_at is not None

    with session_scope(factory) as s:
        assert job_service.claim_next(s, lease_seconds=600) is None  # nothing left queued


def test_fail_retries_then_fails(factory):
    with session_scope(factory) as s:
        user = _user(s)
        job = job_service.enqueue(s, user.id, JobType.ANALYZE, max_attempts=2)
        job_id = job.id

    # attempt 1 -> fail -> requeued with backoff
    with session_scope(factory) as s:
        job = job_service.claim_next(s, 600)
        job_service.fail(s, job, "boom")
        assert job.status is JobStatus.QUEUED
        assert job.scheduled_at.replace(tzinfo=timezone.utc) > datetime.now(timezone.utc)

    # make it eligible again, attempt 2 -> fail -> terminal FAILED
    with session_scope(factory) as s:
        job = s.get(Job, job_id)
        job.scheduled_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    with session_scope(factory) as s:
        job = job_service.claim_next(s, 600)
        assert job.attempts == 2
        job_service.fail(s, job, "boom again")
        assert job.status is JobStatus.FAILED


def test_reclaim_stale(factory):
    with session_scope(factory) as s:
        user = _user(s)
        job = job_service.enqueue(s, user.id, JobType.ANALYZE)
        job = job_service.claim_next(s, 600)
        job.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)  # dead worker

    with session_scope(factory) as s:
        assert job_service.reclaim_stale(s) == 1

    with session_scope(factory) as s:
        again = job_service.claim_next(s, 600)
        assert again is not None  # reclaimed and runnable


def test_complete_and_cancel(factory):
    with session_scope(factory) as s:
        user = _user(s)
        a = job_service.enqueue(s, user.id, JobType.ANALYZE)
        b = job_service.enqueue(s, user.id, JobType.FIX)
        a_id, b_id = a.id, b.id

    with session_scope(factory) as s:
        a = job_service.claim_next(s, 600)
        job_service.complete(s, a, {"issues": 3}, run_id=None)
        assert a.status is JobStatus.SUCCEEDED
        assert a.progress == 1.0
        assert a.result == {"issues": 3}

    with session_scope(factory) as s:
        b = s.get(Job, b_id)
        job_service.cancel(s, b)
        assert b.status is JobStatus.CANCELED
