"""FastAPI dependencies.

Singletons (engine, infra adapters) live on ``app.state`` so tests can build an
app with overridden settings/fakes. Per-request dependencies (db session,
current user) are resolved here.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, Header, Request
from sqlalchemy.orm import Session

from .config import Settings
from .errors import Unauthorized
from .models import User
from .services.auth import user_from_access_token


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_session(request: Request) -> Iterator[Session]:
    session: Session = request.app.state.session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_blob_store(request: Request):
    return request.app.state.blob_store


def get_email_sender(request: Request):
    return request.app.state.email_sender


def get_embedder(request: Request):
    return request.app.state.embedder


def get_extractor(request: Request):
    return request.app.state.extractor


def get_verifier(request: Request):
    return request.app.state.verifier


def get_billing(request: Request):
    return request.app.state.billing


def get_worker(request: Request):
    return request.app.state.worker


def get_current_user(
    authorization: str | None = Header(default=None),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise Unauthorized("missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    return user_from_access_token(session, settings, token)
