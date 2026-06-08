"""Billing: plan resolution and Stripe integration (behind an interface).

``current_plan`` is the single source of truth for what a user may do. The
``BillingProvider`` abstracts Stripe so dev runs with billing disabled while prod
creates real Checkout sessions and processes webhooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings
from ..errors import AppError, NotFound
from ..models import StripeEvent, Subscription, SubscriptionStatus, User
from ..plans import PlanTier

# Stripe statuses that grant access to the paid plan.
_ACTIVE_STATUSES = {SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING}


def current_plan(session: Session, user_id: str) -> PlanTier:
    sub = session.scalar(select(Subscription).where(Subscription.user_id == user_id))
    if sub and sub.status in _ACTIVE_STATUSES:
        return sub.plan
    return PlanTier.FREE


def get_or_create_subscription(session: Session, user_id: str) -> Subscription:
    sub = session.scalar(select(Subscription).where(Subscription.user_id == user_id))
    if sub is None:
        sub = Subscription(user_id=user_id, plan=PlanTier.FREE, status=SubscriptionStatus.ACTIVE)
        session.add(sub)
        session.flush()
    return sub


@dataclass
class CheckoutSession:
    url: str
    id: str


class BillingProvider(ABC):
    @abstractmethod
    def create_checkout(self, session: Session, user: User, tier: PlanTier) -> CheckoutSession: ...

    @abstractmethod
    def create_portal(self, session: Session, user: User) -> str: ...

    @abstractmethod
    def handle_webhook(self, session: Session, payload: bytes, signature: str) -> str: ...


class DisabledBilling(BillingProvider):
    """Dev: no Stripe. Upgrades flip the plan directly so flows are testable."""

    def create_checkout(self, session: Session, user: User, tier: PlanTier) -> CheckoutSession:
        sub = get_or_create_subscription(session, user.id)
        sub.plan = tier
        sub.status = SubscriptionStatus.ACTIVE
        return CheckoutSession(url=f"/billing/success?tier={tier.value}", id="dev_checkout")

    def create_portal(self, session: Session, user: User) -> str:
        return "/billing"

    def handle_webhook(self, session: Session, payload: bytes, signature: str) -> str:
        return "ignored (billing disabled)"


class StripeBilling(BillingProvider):
    """Prod: real Stripe Checkout + customer portal + webhook processing."""

    def __init__(self, settings: Settings):
        import stripe  # lazy

        self.settings = settings
        self.stripe = stripe
        stripe.api_key = settings.stripe_secret_key
        self._price = {
            PlanTier.BASIC: settings.stripe_price_basic,
            PlanTier.PRO: settings.stripe_price_pro,
        }

    def _ensure_customer(self, session: Session, user: User) -> str:
        if user.stripe_customer_id:
            return user.stripe_customer_id
        customer = self.stripe.Customer.create(email=user.email, metadata={"user_id": user.id})
        user.stripe_customer_id = customer["id"]
        session.flush()
        return customer["id"]

    def create_checkout(self, session: Session, user: User, tier: PlanTier) -> CheckoutSession:
        price = self._price.get(tier)
        if not price:
            raise AppError(f"no Stripe price configured for plan {tier.value}")
        customer = self._ensure_customer(session, user)
        cs = self.stripe.checkout.Session.create(
            mode="subscription",
            customer=customer,
            line_items=[{"price": price, "quantity": 1}],
            success_url=f"{self.settings.frontend_url}/billing/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{self.settings.frontend_url}/billing",
            metadata={"user_id": user.id, "tier": tier.value},
        )
        return CheckoutSession(url=cs["url"], id=cs["id"])

    def create_portal(self, session: Session, user: User) -> str:
        customer = self._ensure_customer(session, user)
        portal = self.stripe.billing_portal.Session.create(
            customer=customer, return_url=f"{self.settings.frontend_url}/billing"
        )
        return portal["url"]

    def handle_webhook(self, session: Session, payload: bytes, signature: str) -> str:
        event = self.stripe.Webhook.construct_event(
            payload, signature, self.settings.stripe_webhook_secret
        )
        # Idempotency: skip events we've already processed.
        if session.scalar(select(StripeEvent).where(StripeEvent.event_id == event["id"])):
            return "duplicate"
        session.add(StripeEvent(event_id=event["id"], type=event["type"], payload=dict(event)))

        handler = getattr(self, f"_on_{event['type'].replace('.', '_')}", None)
        if handler:
            handler(session, event["data"]["object"])
        # mark processed
        rec = session.scalar(select(StripeEvent).where(StripeEvent.event_id == event["id"]))
        rec.processed_at = datetime.now(timezone.utc)
        return event["type"]

    # --- webhook handlers ---

    def _sub_for_customer(self, session: Session, customer_id: str) -> Subscription:
        user = session.scalar(select(User).where(User.stripe_customer_id == customer_id))
        if user is None:
            raise NotFound(f"no user for stripe customer {customer_id}")
        return get_or_create_subscription(session, user.id)

    def _tier_for_price(self, price_id: str) -> PlanTier:
        for tier, pid in self._price.items():
            if pid == price_id:
                return tier
        return PlanTier.FREE

    def _on_checkout_session_completed(self, session: Session, obj: dict) -> None:
        tier = PlanTier(obj.get("metadata", {}).get("tier", "basic"))
        sub = self._sub_for_customer(session, obj["customer"])
        sub.plan = tier
        sub.status = SubscriptionStatus.ACTIVE
        sub.stripe_subscription_id = obj.get("subscription")

    def _apply_subscription(self, session: Session, obj: dict) -> None:
        sub = self._sub_for_customer(session, obj["customer"])
        sub.stripe_subscription_id = obj["id"]
        sub.status = _map_status(obj.get("status", "active"))
        items = obj.get("items", {}).get("data", [])
        if items:
            price_id = items[0]["price"]["id"]
            sub.stripe_price_id = price_id
            sub.plan = self._tier_for_price(price_id)
        sub.cancel_at_period_end = bool(obj.get("cancel_at_period_end"))
        if obj.get("current_period_end"):
            sub.current_period_end = datetime.fromtimestamp(obj["current_period_end"], tz=timezone.utc)

    _on_customer_subscription_created = _apply_subscription
    _on_customer_subscription_updated = _apply_subscription

    def _on_customer_subscription_deleted(self, session: Session, obj: dict) -> None:
        sub = self._sub_for_customer(session, obj["customer"])
        sub.status = SubscriptionStatus.CANCELED
        sub.plan = PlanTier.FREE


def _map_status(stripe_status: str) -> SubscriptionStatus:
    mapping = {
        "active": SubscriptionStatus.ACTIVE,
        "trialing": SubscriptionStatus.TRIALING,
        "past_due": SubscriptionStatus.PAST_DUE,
        "canceled": SubscriptionStatus.CANCELED,
        "unpaid": SubscriptionStatus.PAST_DUE,
        "incomplete": SubscriptionStatus.INCOMPLETE,
        "incomplete_expired": SubscriptionStatus.CANCELED,
    }
    return mapping.get(stripe_status, SubscriptionStatus.INCOMPLETE)


def get_billing_provider(settings: Settings) -> BillingProvider:
    if settings.billing_backend == "stripe":
        return StripeBilling(settings)
    return DisabledBilling()
