"""Auth routes: register, verify, login, refresh, logout, password reset."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import schemas
from ..config import Settings
from ..deps import get_current_user, get_email_sender, get_session, get_settings
from ..models import User
from ..services import auth as auth_service
from ..services.billing import current_plan

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=schemas.MessageResponse, status_code=201)
def register(
    body: schemas.RegisterRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    email_sender=Depends(get_email_sender),
):
    auth_service.register(session, settings, email_sender, body.email, body.password)
    return {"message": "registered; check your email for a verification code"}


@router.post("/verify-email", response_model=schemas.MessageResponse)
def verify_email(body: schemas.VerifyEmailRequest, session: Session = Depends(get_session)):
    auth_service.verify_email(session, body.email, body.pin)
    return {"message": "email verified"}


@router.post("/resend-pin", response_model=schemas.MessageResponse)
def resend_pin(
    body: schemas.ResendPinRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    email_sender=Depends(get_email_sender),
):
    from sqlalchemy import select

    user = session.scalar(select(User).where(User.email == body.email.lower()))
    if user and not user.email_verified:
        auth_service.issue_email_pin(session, settings, email_sender, user)
    return {"message": "if the account exists and is unverified, a new code was sent"}


@router.post("/login", response_model=schemas.TokenResponse)
def login(
    body: schemas.LoginRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
):
    pair = auth_service.login(session, settings, body.email, body.password)
    return schemas.TokenResponse(access_token=pair.access_token, refresh_token=pair.refresh_token)


@router.post("/refresh", response_model=schemas.TokenResponse)
def refresh(
    body: schemas.RefreshRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
):
    pair = auth_service.refresh_tokens(session, settings, body.refresh_token)
    return schemas.TokenResponse(access_token=pair.access_token, refresh_token=pair.refresh_token)


@router.post("/logout", response_model=schemas.MessageResponse)
def logout(body: schemas.RefreshRequest, session: Session = Depends(get_session)):
    auth_service.logout(session, body.refresh_token)
    return {"message": "logged out"}


@router.post("/forgot-password", response_model=schemas.MessageResponse)
def forgot_password(
    body: schemas.ForgotPasswordRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    email_sender=Depends(get_email_sender),
):
    auth_service.request_password_reset(session, settings, email_sender, body.email)
    return {"message": "if the account exists, a reset link was sent"}


@router.post("/reset-password", response_model=schemas.MessageResponse)
def reset_password(body: schemas.ResetPasswordRequest, session: Session = Depends(get_session)):
    auth_service.reset_password(session, body.token, body.password)
    return {"message": "password reset; please log in"}


@router.get("/me", response_model=schemas.UserResponse)
def me(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    return schemas.UserResponse(
        id=user.id, email=user.email, email_verified=user.email_verified,
        plan=current_plan(session, user.id).value,
    )
