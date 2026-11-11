"""
PayPal Webhook Receiver Router.

1. Verifies every event with PayPal's official API
   (POST /v1/notifications/verify-webhook-signature). Unverified events are rejected (401).
2. Idempotent: each PayPal event id is processed at most once (PayPal retries deliveries).
3. Reconciles events with our orders:
   - CHECKOUT.ORDER.APPROVED  -> confirm approval (re-checked with PayPal, same as the return URL)
   - PAYMENT.CAPTURE.COMPLETED -> must match a capture made through the risk gate; otherwise flagged
   - PAYMENT.CAPTURE.DENIED / REFUNDED / REVERSED -> recorded on the order
"""

import json
import uuid
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Request, HTTPException
from ..paypal_integration.client import PayPalClient
from ..database import db
from . import payments
from ..core.security import require_merchant_or_analyst

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])
paypal_client = PayPalClient()

MAX_WEBHOOK_BODY_BYTES = 256 * 1024
CAPTURE_STATUS_EVENTS = {
    "PAYMENT.CAPTURE.DENIED": "DENIED",
    "PAYMENT.CAPTURE.REFUNDED": "REFUNDED",
    "PAYMENT.CAPTURE.REVERSED": "REVERSED",
}


def _related_order_id(resource: Dict[str, Any]) -> Optional[str]:
    """Capture resources reference their order under supplementary_data.related_ids.order_id."""
    try:
        return resource["supplementary_data"]["related_ids"]["order_id"]
    except (KeyError, TypeError):
        return None


async def _process_event(event_type: str, resource: Dict[str, Any]) -> Dict[str, str]:
    """Applies a verified event to our order records. Returns {"status", "note"}."""
    if event_type == "CHECKOUT.ORDER.APPROVED":
        order_id = resource.get("id")
        order = db.get_payment_order(order_id) if order_id else None
        if not order:
            return {"status": "IGNORED", "note": "Order not created by PayGuard."}
        if order["status"] != "CREATED":
            return {"status": "PROCESSED", "note": f"Order already {order['status']}; nothing to do."}
        try:
            # Same server-side check as the browser return URL: PayPal API + amount match.
            await payments.verify_buyer_approval(order_id)
            return {"status": "PROCESSED", "note": "Buyer approval confirmed with PayPal API."}
        except HTTPException as exc:
            return {"status": "FAILED", "note": f"Approval not confirmed: {exc.detail}"}

    if event_type == "PAYMENT.CAPTURE.COMPLETED":
        order_id = _related_order_id(resource)
        order = db.get_payment_order(order_id) if order_id else None
        if not order:
            return {"status": "IGNORED", "note": "Capture is not for a PayGuard order."}
        if order["status"] == "CAPTURED" and order.get("capture_id") == resource.get("id"):
            return {"status": "PROCESSED", "note": "Capture reconciled with risk-gated capture."}
        # Money moved without passing through our gate (or ids disagree): flag for review.
        db.log_audit("PAYMENT_ORDER", order_id, "CAPTURE_OUTSIDE_RISK_GATE", {
            "webhook_capture_id": resource.get("id"), "stored_status": order["status"],
            "stored_capture_id": order.get("capture_id")})
        return {"status": "FLAGGED", "note": "Capture not made through the PayGuard risk gate."}

    if event_type in CAPTURE_STATUS_EVENTS:
        order_id = _related_order_id(resource)
        if order_id and db.update_order_paypal_status(order_id, CAPTURE_STATUS_EVENTS[event_type], event_type):
            return {"status": "PROCESSED", "note": f"Order marked {CAPTURE_STATUS_EVENTS[event_type]}."}
        return {"status": "IGNORED", "note": "Capture is not for a PayGuard order."}

    return {"status": "IGNORED", "note": f"No handler for {event_type}; logged only."}


@router.post("/paypal")
async def handle_paypal_webhook(request: Request):
    raw = await request.body()
    if len(raw) > MAX_WEBHOOK_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Webhook body too large.")
    try:
        event_data = json.loads(raw)
        if not isinstance(event_data, dict):
            raise ValueError
    except ValueError:
        raise HTTPException(status_code=400, detail="Webhook body is not a JSON object.")

    event_id = str(event_data.get("id") or "")
    event_type = str(event_data.get("event_type") or "UNKNOWN")
    resource = event_data.get("resource") if isinstance(event_data.get("resource"), dict) else {}

    # 1. Signature verification with PayPal. Nothing below runs for unverified events.
    verify_result = await paypal_client.verify_webhook_signature(dict(request.headers), event_data)
    if verify_result.get("verification_status") != "SUCCESS":
        reason = verify_result.get("reason", "Invalid signature")
        # Stored under our own id, never the sender-claimed id, so a forged event
        # cannot pre-occupy a real PayPal event id.
        db.log_payment_event(
            event_id=f"REJECTED-{uuid.uuid4().hex[:12]}",
            order_id=str(resource.get("id", "unknown"))[:64],
            event_type=event_type[:80],
            status="SIGNATURE_REJECTED",
            details={"claimed_event_id": event_id[:80], "reason": reason},
            verification_status="FAILURE",
        )
        raise HTTPException(status_code=401, detail=f"PayPal webhook signature verification failed: {reason}")

    if not event_id:
        raise HTTPException(status_code=400, detail="Verified event has no id.")

    # 2. Idempotency: PayPal may deliver the same event several times.
    related_order = resource.get("id") if event_type.startswith("CHECKOUT.") else _related_order_id(resource)
    if not db.claim_webhook_event(event_id, related_order or "unknown", event_type, event_data):
        return {"status": "DUPLICATE_IGNORED", "event_id": event_id, "event_type": event_type}

    # 3. Reconcile with our orders.
    try:
        result = await _process_event(event_type, resource)
    except Exception:
        logger.exception("Webhook processing failed for %s", event_id)
        result = {"status": "FAILED", "note": "Internal processing error."}
    db.set_webhook_event_result(event_id, result["status"], result["note"])

    return {"status": result["status"], "verification_status": "SUCCESS",
            "event_id": event_id, "event_type": event_type, "note": result["note"]}


@router.get("/paypal/events", dependencies=[Depends(require_merchant_or_analyst)])
async def list_webhook_events(limit: int = 50):
    """List recent payment/webhook events from the database."""
    return db.list_payment_events(limit=min(max(limit, 1), 200))
