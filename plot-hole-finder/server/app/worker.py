"""Background worker: claims jobs and runs the analysis pipeline.

Deployment-agnostic. The same class powers three modes (see config.job_runner):
- inline   : ``run_available`` drains the queue in-process (tests / simple dev)
- thread   : a daemon thread runs ``run_forever`` inside the API (dev)
- external : ``python -m app.worker`` runs ``run_forever`` as its own process
             (prod; scale by running more replicas -- e.g. a K8s Deployment
             autoscaled on queue depth)

Claiming is short and lease-based; the long pipeline runs outside any row lock,
heartbeating to extend its lease so a healthy job is never reclaimed.
"""

from __future__ import annotations

import logging
import time

from sqlalchemy.orm import Session, sessionmaker

from .config import Settings, get_settings
from .db import auto_migrate, init_db, make_engine, make_session_factory, session_scope
from .embeddings import get_embedder
from .llm import get_extractor, get_verifier
from .models import Job, JobStatus, JobType, Project, User
from .services import analysis as analysis_service
from .services import jobs as job_service
from .storage.blob import get_blob_store

log = logging.getLogger("plothole.worker")


class Worker:
    def __init__(self, session_factory: sessionmaker[Session], settings: Settings,
                 blob_store, embedder, extractor, verifier):
        self.sf = session_factory
        self.settings = settings
        self.blob_store = blob_store
        self.embedder = embedder
        self.extractor = extractor
        self.verifier = verifier
        self.lease = settings.job_lease_seconds
        self._stop = False

    # --- public API ---

    def run_available(self, limit: int = 1000) -> int:
        """Claim and run queued jobs until none remain (or limit). Returns count."""
        processed = 0
        while processed < limit:
            job_id = self._claim()
            if job_id is None:
                break
            self._run(job_id)
            processed += 1
        return processed

    def run_forever(self) -> None:
        log.info("worker started (poll=%ss, lease=%ss)", self.settings.job_poll_interval, self.lease)
        while not self._stop:
            with session_scope(self.sf) as s:
                job_service.reclaim_stale(s)
            if self.run_available() == 0:
                time.sleep(self.settings.job_poll_interval)

    def stop(self) -> None:
        self._stop = True

    # --- internals ---

    def _claim(self) -> str | None:
        with session_scope(self.sf) as s:
            job = job_service.claim_next(s, self.lease)
            return job.id if job else None

    def _run(self, job_id: str) -> None:
        try:
            with session_scope(self.sf) as s:
                job = s.get(Job, job_id)
                if job is None or job.status != JobStatus.RUNNING:
                    return
                handler = self._handlers()[job.type]
                result, run_id = handler(s, job)
                job_service.complete(s, job, result, run_id)
        except Exception as exc:  # mark failed/retry in a fresh session
            log.exception("job %s failed", job_id)
            with session_scope(self.sf) as s2:
                job = s2.get(Job, job_id)
                if job:
                    job_service.fail(s2, job, repr(exc))

    def _handlers(self):
        return {JobType.ANALYZE: self._handle_analyze, JobType.FIX: self._handle_fix}

    def _heartbeat(self, session: Session, job: Job):
        def cb(step: str, fraction: float) -> None:
            job_service.heartbeat(session, job, self.lease, fraction, step)
        return cb

    def _handle_analyze(self, session: Session, job: Job):
        project = session.get(Project, job.project_id)
        user = session.get(User, job.user_id)
        run = analysis_service.run_find(
            session, self.settings, self.blob_store, self.embedder, self.extractor,
            self.verifier, project, user, on_progress=self._heartbeat(session, job),
        )
        return {"run_id": run.id, **(run.stats or {})}, run.id

    def _handle_fix(self, session: Session, job: Job):
        project = session.get(Project, job.project_id)
        user = session.get(User, job.user_id)
        run = analysis_service.run_fix(
            session, self.settings, self.embedder, self.verifier, project, user
        )
        return {"run_id": run.id, **(run.stats or {})}, run.id


def build_worker(settings: Settings, session_factory: sessionmaker[Session]) -> Worker:
    return Worker(
        session_factory, settings,
        blob_store=get_blob_store(settings),
        embedder=get_embedder(settings),
        extractor=get_extractor(settings),
        verifier=get_verifier(settings),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    engine = make_engine(settings)
    auto_migrate(engine) if settings.auto_migrate else init_db(engine)
    worker = build_worker(settings, make_session_factory(engine))
    worker.run_forever()


if __name__ == "__main__":
    main()
