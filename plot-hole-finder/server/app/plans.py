"""Plan catalog and limits (code-defined; the DB records a user's current tier).

Three tiers:

- FREE  -- a taste: one small document, find-only. Big enough to prove it works.
- BASIC -- real documents, find-only.
- PRO   -- large documents + "fix this for me", included in the subscription
           within a monthly fix quota (overage is out of scope for now).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class PlanTier(str, enum.Enum):
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"


@dataclass(frozen=True)
class PlanLimits:
    tier: PlanTier
    max_projects: int
    max_words_per_project: int
    can_find: bool
    can_fix: bool
    fixes_per_month: int
    # Stripe price id is wired from env at runtime (None for FREE / local dev).
    stripe_price_env: str | None = None


PLANS: dict[PlanTier, PlanLimits] = {
    PlanTier.FREE: PlanLimits(
        tier=PlanTier.FREE,
        max_projects=1,
        max_words_per_project=3_000,
        can_find=True,
        can_fix=False,
        fixes_per_month=0,
    ),
    PlanTier.BASIC: PlanLimits(
        tier=PlanTier.BASIC,
        max_projects=20,
        max_words_per_project=300_000,
        can_find=True,
        can_fix=False,
        fixes_per_month=0,
        stripe_price_env="stripe_price_basic",
    ),
    PlanTier.PRO: PlanLimits(
        tier=PlanTier.PRO,
        max_projects=200,
        max_words_per_project=600_000,
        can_find=True,
        can_fix=True,
        fixes_per_month=50,
        stripe_price_env="stripe_price_pro",
    ),
}


def limits_for(tier: PlanTier) -> PlanLimits:
    return PLANS[tier]
