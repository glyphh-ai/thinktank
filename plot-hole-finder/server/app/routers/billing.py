"""Billing routes: checkout, customer portal, and the Stripe webhook."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Request
from sqlalchemy.orm import Session

from .. import schemas
from ..deps import get_billing, get_current_user, get_session
from ..errors import ValidationError
from ..models import User
from ..plans import PlanTier

router = APIRouter(prefix="/billing", tags=["billing"])


@router.post("/checkout", response_model=schemas.CheckoutResponse)
def checkout(
    body: schemas.CheckoutRequest,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
    billing=Depends(get_billing),
):
    try:
        tier = PlanTier(body.tier)
    except ValueError as exc:
        raise ValidationError(f"unknown plan tier: {body.tier}") from exc
    if tier == PlanTier.FREE:
        raise ValidationError("cannot checkout the free plan")
    cs = billing.create_checkout(session, user, tier)
    return schemas.CheckoutResponse(url=cs.url)


@router.post("/portal", response_model=schemas.CheckoutResponse)
def portal(
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
    billing=Depends(get_billing),
):
    return schemas.CheckoutResponse(url=billing.create_portal(session, user))


@router.post("/webhook", response_model=schemas.MessageResponse)
async def webhook(
    request: Request,
    stripe_signature: str | None = Header(default=None),
    session: Session = Depends(get_session),
    billing=Depends(get_billing),
):
    payload = await request.body()
    result = billing.handle_webhook(session, payload, stripe_signature or "")
    return {"message": result}
