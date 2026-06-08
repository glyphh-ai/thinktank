"""User accounts and auth-related tokens.

The storage layer holds *hashes* only -- never plaintext passwords or PINs. The
auth service (next layer up) is responsible for hashing/verifying; these models
just give it durable, swappable homes.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, PrimaryKey, Timestamps


class UserStatus(str, enum.Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"


class TokenKind(str, enum.Enum):
    EMAIL_PIN = "email_pin"          # short numeric code, emailed for verification
    PASSWORD_RESET = "password_reset"  # long opaque token for reset links


class User(Base, PrimaryKey, Timestamps):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[UserStatus] = mapped_column(
        Enum(UserStatus, native_enum=False, length=16), default=UserStatus.ACTIVE, nullable=False
    )
    # Set once a Stripe Customer is created for this user.
    stripe_customer_id: Mapped[str | None] = mapped_column(String(64), index=True)

    tokens: Mapped[list["VerificationToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class VerificationToken(Base, PrimaryKey, Timestamps):
    """Email PIN verification and password-reset tokens (hashed at rest)."""

    __tablename__ = "verification_tokens"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[TokenKind] = mapped_column(Enum(TokenKind, native_enum=False, length=20), nullable=False)
    secret_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # brute-force guard

    user: Mapped[User] = relationship(back_populates="tokens")


class RefreshToken(Base, PrimaryKey, Timestamps):
    """Persisted refresh tokens (hashed) for JWT rotation/revocation.

    Access tokens are stateless JWTs and are not stored.
    """

    __tablename__ = "refresh_tokens"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    user_agent: Mapped[str | None] = mapped_column(String(255))
    ip_address: Mapped[str | None] = mapped_column(String(64))

    user: Mapped[User] = relationship(back_populates="refresh_tokens")
