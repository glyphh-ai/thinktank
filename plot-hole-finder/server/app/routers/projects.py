"""Project routes: create (paste/upload), list, get, delete."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from .. import schemas
from ..deps import get_blob_store, get_current_user, get_session
from ..models import SourceType, User
from ..services import projects as project_service

router = APIRouter(prefix="/projects", tags=["projects"])


def _to_response(p) -> schemas.ProjectResponse:
    return schemas.ProjectResponse(
        id=p.id, name=p.name, status=p.status.value, word_count=p.word_count, created_at=p.created_at
    )


@router.post("", response_model=schemas.ProjectResponse, status_code=201)
def create_from_text(
    body: schemas.CreateProjectRequest,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
    blob_store=Depends(get_blob_store),
):
    project = project_service.create_project(
        session, blob_store, user, body.name, SourceType.PASTE, text=body.text
    )
    return _to_response(project)


@router.post("/upload", response_model=schemas.ProjectResponse, status_code=201)
async def create_from_upload(
    name: str = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
    blob_store=Depends(get_blob_store),
):
    data = await file.read()
    project = project_service.create_project(
        session, blob_store, user, name, SourceType.UPLOAD,
        file_bytes=data, filename=file.filename,
    )
    return _to_response(project)


@router.get("", response_model=list[schemas.ProjectResponse])
def list_projects(session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    return [_to_response(p) for p in project_service.list_projects(session, user)]


@router.get("/{project_id}", response_model=schemas.ProjectResponse)
def get_project(
    project_id: str, session: Session = Depends(get_session), user: User = Depends(get_current_user)
):
    return _to_response(project_service.get_project(session, user, project_id))


@router.delete("/{project_id}", response_model=schemas.MessageResponse)
def delete_project(
    project_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
    blob_store=Depends(get_blob_store),
):
    project_service.delete_project(session, blob_store, user, project_id)
    return {"message": "deleted"}
