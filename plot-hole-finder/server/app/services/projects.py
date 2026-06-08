"""Projects: create from upload/paste, list, fetch, delete -- with plan limits."""

from __future__ import annotations

import hashlib

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..errors import NotFound, PlanLimitExceeded, ValidationError
from ..models import (
    Document,
    Project,
    ProjectStatus,
    SourceType,
    UsageEvent,
    UsageMetric,
    User,
)
from ..parsing import extract_text, mime_for
from ..plans import limits_for
from ..storage.blob import BlobStore
from ..text import word_count
from .billing import current_plan


def _blob_key(user_id: str, project_id: str, document_id: str) -> str:
    return f"users/{user_id}/projects/{project_id}/documents/{document_id}"


def create_project(
    session: Session,
    blob_store: BlobStore,
    user: User,
    name: str,
    source_type: SourceType,
    *,
    text: str | None = None,
    file_bytes: bytes | None = None,
    filename: str | None = None,
) -> Project:
    limits = limits_for(current_plan(session, user.id))

    project_count = session.scalar(
        select(func.count()).select_from(Project).where(Project.user_id == user.id)
    )
    if project_count >= limits.max_projects:
        raise PlanLimitExceeded(
            f"plan limit reached: {limits.max_projects} project(s). Upgrade to add more."
        )

    if source_type == SourceType.PASTE:
        if not text or not text.strip():
            raise ValidationError("pasted text is empty")
        data = text.encode("utf-8")
        filename = filename or "pasted.txt"
        mime = "text/plain"
        body_text = text
    else:
        if not file_bytes:
            raise ValidationError("uploaded file is empty")
        data = file_bytes
        filename = filename or "upload.txt"
        mime = mime_for(filename)
        body_text = extract_text(filename, data)  # parse PDF/Word -> plain text

    if not body_text.strip():
        raise ValidationError("no readable text could be extracted from the document")
    wc = word_count(body_text)
    if wc > limits.max_words_per_project:
        raise PlanLimitExceeded(
            f"document has {wc:,} words; your plan allows {limits.max_words_per_project:,}. "
            "Upgrade for larger documents."
        )

    project = Project(user_id=user.id, name=name.strip() or "Untitled", status=ProjectStatus.CREATED,
                      word_count=wc)
    session.add(project)
    session.flush()

    document = Document(
        project_id=project.id, filename=filename, mime_type=mime, source_type=source_type,
        blob_key="", byte_size=len(data), word_count=wc, sha256=hashlib.sha256(data).hexdigest(),
    )
    session.add(document)
    session.flush()

    key = _blob_key(user.id, project.id, document.id)
    blob_store.put(key, data, mime)
    document.blob_key = key
    if source_type == SourceType.PASTE:
        document.text_blob_key = key  # already plain text
    else:
        text_key = f"{key}.txt"  # keep the original bytes; store parsed text alongside
        blob_store.put(text_key, body_text.encode("utf-8"), "text/plain")
        document.text_blob_key = text_key

    session.add(UsageEvent(user_id=user.id, project_id=project.id,
                           metric=UsageMetric.PROJECTS_CREATED, amount=1))
    return project


def list_projects(session: Session, user: User) -> list[Project]:
    return list(session.scalars(
        select(Project).where(Project.user_id == user.id).order_by(Project.created_at.desc())
    ))


def get_project(session: Session, user: User, project_id: str) -> Project:
    project = session.get(Project, project_id)
    if project is None or project.user_id != user.id:
        raise NotFound("project not found")
    return project


def delete_project(session: Session, blob_store: BlobStore, user: User, project_id: str) -> None:
    project = get_project(session, user, project_id)
    for doc in project.documents:
        if doc.blob_key:
            blob_store.delete(doc.blob_key)
    session.delete(project)


def document_text(blob_store: BlobStore, document: Document) -> str:
    # Prefer the parsed-text blob; fall back to the original (e.g. legacy rows).
    key = document.text_blob_key or document.blob_key
    return blob_store.get(key).decode("utf-8", errors="replace")
