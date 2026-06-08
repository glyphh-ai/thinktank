"""Analysis routes: enqueue find/fix jobs and read results."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import schemas
from ..config import Settings
from ..deps import get_current_user, get_session, get_settings, get_worker
from ..errors import Forbidden
from ..models import AnalysisRun, Issue, JobType, User
from ..plans import limits_for
from ..services import jobs as job_service
from ..services import projects as project_service
from ..services.billing import current_plan

router = APIRouter(prefix="/projects/{project_id}", tags=["analysis"])


def _job_response(job) -> schemas.JobResponse:
    return schemas.JobResponse(
        id=job.id, type=job.type.value, status=job.status.value, progress=job.progress,
        step=job.step, error=job.error, result=job.result, run_id=job.run_id,
    )


def _enqueue(session, settings: Settings, worker, user, project_id, job_type) -> object:
    job = job_service.enqueue(session, user.id, job_type, project_id=project_id,
                              max_attempts=settings.job_max_attempts)
    if settings.job_runner == "inline":
        # Tests / simple dev: process synchronously so callers can poll right away.
        session.commit()
        worker.run_available()
        session.refresh(job)
    return job


@router.post("/analyze", response_model=schemas.JobResponse, status_code=202)
def analyze(
    project_id: str,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    user: User = Depends(get_current_user),
    worker=Depends(get_worker),
):
    project = project_service.get_project(session, user, project_id)
    job = _enqueue(session, settings, worker, user, project.id, JobType.ANALYZE)
    return _job_response(job)


@router.post("/fix", response_model=schemas.JobResponse, status_code=202)
def fix(
    project_id: str,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    user: User = Depends(get_current_user),
    worker=Depends(get_worker),
):
    project = project_service.get_project(session, user, project_id)
    if not limits_for(current_plan(session, user.id)).can_fix:
        raise Forbidden("fix-this-for-me requires the Pro plan")
    job = _enqueue(session, settings, worker, user, project.id, JobType.FIX)
    return _job_response(job)


@router.get("/issues", response_model=list[schemas.IssueResponse])
def list_issues(
    project_id: str, session: Session = Depends(get_session), user: User = Depends(get_current_user)
):
    project = project_service.get_project(session, user, project_id)
    issues = session.scalars(
        select(Issue).where(Issue.project_id == project.id).order_by(Issue.severity.desc())
    ).all()
    return [
        schemas.IssueResponse(
            id=i.id, category=i.category.value, severity=i.severity.value, status=i.status.value,
            title=i.title, description=i.description, confidence=i.confidence,
            suggested_fix=i.suggested_fix,
            evidence=[
                schemas.EvidenceResponse(role=e.role.value, char_start=e.char_start,
                                         char_end=e.char_end, quote=e.quote)
                for e in i.evidence
            ],
        )
        for i in issues
    ]


@router.get("/runs", response_model=list[schemas.RunResponse])
def list_runs(
    project_id: str, session: Session = Depends(get_session), user: User = Depends(get_current_user)
):
    project = project_service.get_project(session, user, project_id)
    runs = session.scalars(
        select(AnalysisRun).where(AnalysisRun.project_id == project.id)
        .order_by(AnalysisRun.created_at.desc())
    ).all()
    return [
        schemas.RunResponse(id=r.id, kind=r.kind.value, status=r.status.value,
                            stats=r.stats or {}, error=r.error)
        for r in runs
    ]
