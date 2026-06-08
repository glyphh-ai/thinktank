"""Usage ledger for plan-limit enforcement.

An append-only event ledger (auditable) rather than mutable counters. Quota
checks sum events within the current billing period -- see ``app.plans`` for the
limits and ``sum_usage`` for the helper.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import Enum, ForeignKey, Integer, String, and_, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from .base import Base, PrimaryKey, Timestamps


class UsageMetric(str, enum.Enum):
    PROJECTS_CREATED = "projects_created"
    WORDS_ANALYZED = "words_analyzed"
    FIXES_USED = "fixes_used"  # Pro: fix-this-for-me invocations


class UsageEvent(Base, PrimaryKey, Timestamps):
    __tablename__ = "usage_events"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id", ondelete="SET NULL"))
    metric: Mapped[UsageMetric] = mapped_column(Enum(UsageMetric, native_enum=False, length=20), index=True)
    amount: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


def sum_usage(
    session: Session, user_id: str, metric: UsageMetric, since: datetime | None = None
) -> int:
    """Total of a usage metric for a user, optionally since a period start."""
    conds = [UsageEvent.user_id == user_id, UsageEvent.metric == metric]
    if since is not None:
        conds.append(UsageEvent.created_at >= since)
    total = session.scalar(select(func.coalesce(func.sum(UsageEvent.amount), 0)).where(and_(*conds)))
    return int(total or 0)
