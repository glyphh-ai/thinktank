"""Analysis routes: run find / fix and read results."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import schemas
from ..config import Settings
from ..deps import (
    get_blob_store,
    get_current_user,
    get_embedder,
    get_extractor,
    get_session,
    get_settings,
    get_verifier,
)
from ..models import AnalysisRun, Issue, User
from ..services import analysis as analysis_service
from ..services import projects as project_service

router = APIRouter(prefix="/projects/{project_id}", tags=["analysis"])


def _run_response(run: AnalysisRun) -> schemas.RunResponse:
    return schemas.RunResponse(
        id=run.id, kind=run.kind.value, status=run.status.value,
        stats=run.stats or {}, error=run.error,
    )


@router.post("/analyze", response_model=schemas.RunResponse)
def analyze(
    project_id: str,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    user: User = Depends(get_current_user),
    blob_store=Depends(get_blob_store),
    embedder=Depends(get_embedder),
    extractor=Depends(get_extractor),
    verifier=Depends(get_verifier),
):
    project = project_service.get_project(session, user, project_id)
    run = analysis_service.run_find(
        session, settings, blob_store, embedder, extractor, verifier, project, user
    )
    return _run_response(run)


@router.post("/fix", response_model=schemas.RunResponse)
def fix(
    project_id: str,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    user: User = Depends(get_current_user),
    embedder=Depends(get_embedder),
    verifier=Depends(get_verifier),
):
    project = project_service.get_project(session, user, project_id)
    run = analysis_service.run_fix(session, settings, embedder, verifier, project, user)
    return _run_response(run)


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
    return [_run_response(r) for r in runs]
