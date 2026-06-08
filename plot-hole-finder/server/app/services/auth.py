"""Authentication: registration, email-PIN verification, login/JWT, reset."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings
from ..errors import Conflict, Forbidden, Unauthorized, ValidationError
from ..models import RefreshToken, TokenKind, User, VerificationToken
from ..models.user import UserStatus
from ..security import (
    decode_jwt,
    encode_jwt,
    generate_pin,
    generate_token,
    hash_password,
    hash_secret,
    verify_password,
    verify_secret,
)
from ..storage.email import EmailSender, password_reset_message, verification_pin_message
from .billing import get_or_create_subscription

_MAX_PIN_ATTEMPTS = 5


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _expired(dt: datetime) -> bool:
    # SQLite round-trips datetimes as naive; coerce to UTC before comparing.
    aware = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return aware < _now()


def _normalize_email(email: str) -> str:
    email = email.strip().lower()
    if "@" not in email or "." not in email.split("@")[-1]:
        raise ValidationError("invalid email address")
    return email


def register(
    session: Session, settings: Settings, email_sender: EmailSender, email: str, password: str
) -> User:
    email = _normalize_email(email)
    if len(password) < 8:
        raise ValidationError("password must be at least 8 characters")
    if session.scalar(select(User).where(User.email == email)):
        raise Conflict("an account with this email already exists")

    user = User(email=email, password_hash=hash_password(password), email_verified=False)
    session.add(user)
    session.flush()
    get_or_create_subscription(session, user.id)  # starts on FREE
    issue_email_pin(session, settings, email_sender, user)
    return user


def issue_email_pin(
    session: Session, settings: Settings, email_sender: EmailSender, user: User
) -> None:
    pin = generate_pin()
    session.add(VerificationToken(
        user_id=user.id, kind=TokenKind.EMAIL_PIN, secret_hash=hash_secret(pin),
        expires_at=_now() + timedelta(seconds=settings.email_pin_ttl_seconds),
    ))
    email_sender.send(verification_pin_message(settings, user.email, pin))


def verify_email(session: Session, email: str, pin: str) -> User:
    email = _normalize_email(email)
    user = session.scalar(select(User).where(User.email == email))
    if user is None:
        raise Unauthorized("invalid code")
    if user.email_verified:
        return user

    token = session.scalar(
        select(VerificationToken)
        .where(
            VerificationToken.user_id == user.id,
            VerificationToken.kind == TokenKind.EMAIL_PIN,
            VerificationToken.consumed_at.is_(None),
        )
        .order_by(VerificationToken.created_at.desc())
    )
    if token is None or _expired(token.expires_at):
        raise Unauthorized("code expired or not found")
    if token.attempts >= _MAX_PIN_ATTEMPTS:
        raise Forbidden("too many attempts; request a new code")

    token.attempts += 1
    if not verify_secret(pin, token.secret_hash):
        raise Unauthorized("invalid code")

    token.consumed_at = _now()
    user.email_verified = True
    return user


def login(session: Session, settings: Settings, email: str, password: str) -> TokenPair:
    email = _normalize_email(email)
    user = session.scalar(select(User).where(User.email == email))
    if user is None or not verify_password(password, user.password_hash):
        raise Unauthorized("invalid email or password")
    if user.status != UserStatus.ACTIVE:
        raise Forbidden("account is not active")
    if not user.email_verified:
        raise Forbidden("email not verified")
    return issue_tokens(session, settings, user)


def issue_tokens(session: Session, settings: Settings, user: User) -> TokenPair:
    access = encode_jwt(
        {"sub": user.id, "email": user.email}, settings.jwt_secret,
        settings.access_token_ttl_seconds, settings.jwt_algorithm,
    )
    raw_refresh = generate_token()
    session.add(RefreshToken(
        user_id=user.id, token_hash=hash_secret(raw_refresh),
        expires_at=_now() + timedelta(seconds=settings.refresh_token_ttl_seconds),
    ))
    return TokenPair(access_token=access, refresh_token=raw_refresh)


def refresh_tokens(session: Session, settings: Settings, raw_refresh: str) -> TokenPair:
    record = session.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == hash_secret(raw_refresh))
    )
    if record is None or record.revoked_at is not None or _expired(record.expires_at):
        raise Unauthorized("invalid refresh token")
    record.revoked_at = _now()  # rotate
    user = session.get(User, record.user_id)
    if user is None:
        raise Unauthorized("invalid refresh token")
    return issue_tokens(session, settings, user)


def logout(session: Session, raw_refresh: str) -> None:
    record = session.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == hash_secret(raw_refresh))
    )
    if record and record.revoked_at is None:
        record.revoked_at = _now()


def request_password_reset(
    session: Session, settings: Settings, email_sender: EmailSender, email: str
) -> None:
    email = _normalize_email(email)
    user = session.scalar(select(User).where(User.email == email))
    if user is None:
        return  # don't reveal whether the account exists
    raw = generate_token()
    session.add(VerificationToken(
        user_id=user.id, kind=TokenKind.PASSWORD_RESET, secret_hash=hash_secret(raw),
        expires_at=_now() + timedelta(seconds=settings.password_reset_ttl_seconds),
    ))
    reset_url = f"{settings.frontend_url}/reset-password?token={raw}"
    email_sender.send(password_reset_message(settings, user.email, reset_url))


def reset_password(session: Session, raw_token: str, new_password: str) -> None:
    if len(new_password) < 8:
        raise ValidationError("password must be at least 8 characters")
    token = session.scalar(
        select(VerificationToken).where(
            VerificationToken.kind == TokenKind.PASSWORD_RESET,
            VerificationToken.secret_hash == hash_secret(raw_token),
            VerificationToken.consumed_at.is_(None),
        )
    )
    if token is None or _expired(token.expires_at):
        raise Unauthorized("invalid or expired reset token")
    token.consumed_at = _now()
    user = session.get(User, token.user_id)
    user.password_hash = hash_password(new_password)
    # Revoke all active refresh tokens on password change.
    for rt in session.scalars(
        select(RefreshToken).where(
            RefreshToken.user_id == user.id, RefreshToken.revoked_at.is_(None)
        )
    ):
        rt.revoked_at = _now()


def user_from_access_token(session: Session, settings: Settings, token: str) -> User:
    try:
        payload = decode_jwt(token, settings.jwt_secret, settings.jwt_algorithm)
    except Exception as exc:
        raise Unauthorized("invalid token") from exc
    user = session.get(User, payload.get("sub"))
    if user is None or user.status != UserStatus.ACTIVE:
        raise Unauthorized("invalid token")
    return user
