"""Stripe metered billing.

Flow: a user starts a Checkout session for a metered subscription; Stripe's
webhook activates their account and records the subscription item; each
successful compression records a usage increment (by quantized parameter count)
against that item. All Stripe access is lazy so the service runs without the
`stripe` package when billing is disabled.
"""

from __future__ import annotations

from service.config import Settings
from service.models import User


def _stripe(settings: Settings):
    import stripe

    stripe.api_key = settings.stripe_secret_key
    return stripe


def create_checkout_session(settings: Settings, user: User) -> str:
    """Create a Checkout session for the metered plan; return its URL."""
    stripe = _stripe(settings)
    customer_id = user.stripe_customer_id
    if not customer_id:
        customer = stripe.Customer.create(email=user.email, metadata={"user_id": user.id})
        customer_id = customer.id
    session = stripe.checkout.Session.create(
        mode="subscription",
        customer=customer_id,
        line_items=[{"price": settings.stripe_price_id}],
        success_url=f"{settings.public_base_url}/billing/success",
        cancel_url=f"{settings.public_base_url}/billing/cancel",
        metadata={"user_id": user.id},
    )
    return session.url


def parse_webhook(settings: Settings, payload: bytes, signature: str) -> dict:
    """Verify and parse a Stripe webhook event."""
    stripe = _stripe(settings)
    return stripe.Webhook.construct_event(
        payload, signature, settings.stripe_webhook_secret
    )


def record_usage(settings: Settings, user: User, quantity: int) -> None:
    """Record a metered usage increment against the user's subscription item."""
    if not settings.stripe_enabled or not user.subscription_item_id or quantity <= 0:
        return
    stripe = _stripe(settings)
    stripe.SubscriptionItem.create_usage_record(
        user.subscription_item_id,
        quantity=quantity,
        action="increment",
    )
