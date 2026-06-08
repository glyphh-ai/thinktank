"""Stripe billing: subscriptions and idempotent webhook event log.

Plan tiers and their limits live in ``app.plans`` (code-defined catalog); this
table records the *user's current standing* as reflected by Stripe.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, String, JSON
from sqlalchemy.orm import Mapped, mapped_column

from ..plans import PlanTier
from .base import Base, PrimaryKey, Timestamps


class SubscriptionStatus(str, enum.Enum):
    # Mirrors Stripe subscription statuses we care about.
    ACTIVE = "active"
    TRIALING = "trialing"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    INCOMPLETE = "incomplete"


class Subscription(Base, PrimaryKey, Timestamps):
    __tablename__ = "subscriptions"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    plan: Mapped[PlanTier] = mapped_column(
        Enum(PlanTier, native_enum=False, length=16), default=PlanTier.FREE, nullable=False
    )
    status: Mapped[SubscriptionStatus] = mapped_column(
        Enum(SubscriptionStatus, native_enum=False, length=16),
        default=SubscriptionStatus.ACTIVE,
        nullable=False,
    )
    stripe_subscription_id: Mapped[str | None] = mapped_column(String(64), unique=True, index=True)
    stripe_price_id: Mapped[str | None] = mapped_column(String(64))
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancel_at_period_end: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class StripeEvent(Base, PrimaryKey, Timestamps):
    """Dedupe log for Stripe webhooks -- ``event_id`` is the idempotency key."""

    __tablename__ = "stripe_events"

    event_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
