"""Job status routes: poll progress, list, cancel."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import schemas
from ..deps import get_current_user, get_session
from ..errors import NotFound
from ..models import Job, User
from ..services import jobs as job_service

router = APIRouter(tags=["jobs"])


def _job_response(job: Job) -> schemas.JobResponse:
    return schemas.JobResponse(
        id=job.id, type=job.type.value, status=job.status.value, progress=job.progress,
        step=job.step, error=job.error, result=job.result, run_id=job.run_id,
    )


def _owned(session: Session, user: User, job_id: str) -> Job:
    job = job_service.get_job(session, job_id)
    if job is None or job.user_id != user.id:
        raise NotFound("job not found")
    return job


@router.get("/jobs/{job_id}", response_model=schemas.JobResponse)
def get_job(job_id: str, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    return _job_response(_owned(session, user, job_id))


@router.post("/jobs/{job_id}/cancel", response_model=schemas.JobResponse)
def cancel_job(job_id: str, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    job = _owned(session, user, job_id)
    job_service.cancel(session, job)
    return _job_response(job)


@router.get("/projects/{project_id}/jobs", response_model=list[schemas.JobResponse])
def list_project_jobs(
    project_id: str, session: Session = Depends(get_session), user: User = Depends(get_current_user)
):
    jobs = session.scalars(
        select(Job).where(Job.project_id == project_id, Job.user_id == user.id)
        .order_by(Job.created_at.desc())
    ).all()
    return [_job_response(j) for j in jobs]
