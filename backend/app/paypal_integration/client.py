"""
Layer 6: PayPal Sandbox Client.
Communicates with PayPal Orders v2 REST API in Sandbox.
Flow: create order (intent CAPTURE) -> buyer approves -> PayGuard risk gate -> capture order.
There is deliberately no AUTHORIZE path, so the two PayPal flows can never be mixed.
Also implements official PayPal Webhook Signature Verification:
POST /v1/notifications/verify-webhook-signature.
"""

import httpx
import uuid
from typing import Dict, Any, Optional
from ..config import settings
from ..core.security import security_manager

ORDER_INTENT = "CAPTURE"


class PayPalClient:
    def __init__(self):
        self.client_id = settings.PAYPAL_CLIENT_ID
        self.client_secret = settings.PAYPAL_CLIENT_SECRET
        self.base_url = settings.PAYPAL_API_BASE
        self.webhook_id = settings.PAYPAL_WEBHOOK_ID  # no fallback: verification fails if unset
        self.is_configured = bool(self.client_id and self.client_secret and "your_" not in self.client_id)

    async def get_access_token(self) -> Optional[str]:
        """Fetches OAuth 2.0 bearer token from PayPal Sandbox."""
        if not self.is_configured:
            return "mock_sandbox_access_token"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.base_url}/v1/oauth2/token",
                auth=(self.client_id, self.client_secret),
                data={"grant_type": "client_credentials"},
                headers={"Accept": "application/json"}
            )
            if resp.status_code == 200:
                return resp.json().get("access_token")
            return None

    async def _auth_headers(self, request_id: Optional[str] = None) -> Dict[str, str]:
        token = await self.get_access_token()
        if not token:
            raise RuntimeError("Could not obtain a PayPal access token (check Sandbox credentials).")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        if request_id:
            # PayPal idempotency: a retried request with the same id is not executed twice.
            headers["PayPal-Request-Id"] = request_id
        return headers

    async def create_order(
        self,
        amount: float,
        currency: str = "USD",
        description: str = "PayGuard Payment"
    ) -> Dict[str, Any]:
        """
        Creates a PayPal Order with intent CAPTURE.
        Money only moves when capture_order() is called, which PayGuard does
        only after the buyer approves AND the server-side risk gate permits it.
        """
        return_url = f"{settings.PUBLIC_BASE_URL}/payments/return"
        cancel_url = f"{settings.PUBLIC_BASE_URL}/payments/cancel"

        if not self.is_configured:
            order_id = f"ORDER-SANDBOX-{uuid.uuid4().hex[:8].upper()}"
            return {
                "id": order_id,
                "status": "CREATED",
                "intent": ORDER_INTENT,
                "links": [
                    {
                        # Mock mode: "approving" just hits our own return URL (simulated buyer).
                        "href": f"{return_url}?token={order_id}",
                        "rel": "payer-action",
                        "method": "GET"
                    }
                ],
                "mock_mode": True
            }

        headers = await self._auth_headers(request_id=f"payguard-create-{uuid.uuid4().hex}")
        payload = {
            "intent": ORDER_INTENT,
            "purchase_units": [
                {
                    "amount": {
                        "currency_code": currency,
                        "value": f"{amount:.2f}"
                    },
                    "description": description
                }
            ],
            "payment_source": {
                "paypal": {
                    "experience_context": {
                        "brand_name": "PayGuard Protected Merchant",
                        "shipping_preference": "NO_SHIPPING",
                        # CONTINUE (not PAY_NOW): the buyer approves, PayGuard decides on capture.
                        "user_action": "CONTINUE",
                        "return_url": return_url,
                        "cancel_url": cancel_url,
                    }
                }
            }
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{self.base_url}/v2/checkout/orders", json=payload, headers=headers)
            return resp.json()

    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Fetches the authoritative order state from PayPal (GET /v2/checkout/orders/{id}).
        Used to confirm buyer approval instead of trusting browser query parameters.
        """
        if not self.is_configured:
            raise RuntimeError("get_order requires PayPal credentials; mock orders are approved via simulation.")
        headers = await self._auth_headers()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/v2/checkout/orders/{order_id}", headers=headers)
            data = resp.json()
            data["_http_status"] = resp.status_code
            return data

    async def capture_order(self, order_id: str) -> Dict[str, Any]:
        """
        Captures a buyer-approved CAPTURE-intent order once cleared by the PayGuard risk gate.
        Uses a deterministic PayPal-Request-Id so a retry cannot capture the order twice.
        """
        if not self.is_configured or order_id.startswith("ORDER-SANDBOX"):
            return {
                "id": order_id,
                "status": "COMPLETED",
                "details": "Payment captured successfully in PayPal Sandbox mock mode.",
                "mock_mode": True
            }

        headers = await self._auth_headers(request_id=f"payguard-capture-{order_id}")

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{self.base_url}/v2/checkout/orders/{order_id}/capture", headers=headers)
            return resp.json()

    @property
    def webhook_id_configured(self) -> bool:
        return bool(self.webhook_id) and "your_" not in self.webhook_id

    async def verify_webhook_signature(
        self,
        headers: Dict[str, str],
        event_body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Official PayPal Webhook Signature Verification (postback method).
        Calls POST /v1/notifications/verify-webhook-signature with the transmission headers,
        our registered webhook_id and the event body.

        Returns {"verification_status": "SUCCESS"} ONLY when PayPal itself says SUCCESS.
        There is deliberately no mock/offline bypass: without credentials and a webhook id,
        every webhook is rejected.
        """
        h = security_manager.extract_paypal_webhook_headers(headers)

        if not self.is_configured:
            return {"verification_status": "FAILURE",
                    "reason": "PayPal credentials not configured; webhooks cannot be verified."}
        if not self.webhook_id_configured:
            return {"verification_status": "FAILURE",
                    "reason": "PAYPAL_WEBHOOK_ID is not configured."}
        if not security_manager.validate_webhook_header_presence(h):
            return {"verification_status": "FAILURE",
                    "reason": "Missing required PayPal transmission headers."}
        # SSRF guard: the certificate URL must point at an official PayPal API domain.
        if not security_manager.validate_paypal_cert_url(h["cert_url"]):
            return {"verification_status": "FAILURE",
                    "reason": "Invalid or untrusted certificate URL domain."}

        verify_payload = {
            "auth_algo": h["auth_algo"],
            "cert_url": h["cert_url"],
            "transmission_id": h["transmission_id"],
            "transmission_sig": h["transmission_sig"],
            "transmission_time": h["transmission_time"],
            "webhook_id": self.webhook_id,
            "webhook_event": event_body,
        }

        try:
            verify_headers = await self._auth_headers()
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/notifications/verify-webhook-signature",
                    json=verify_payload,
                    headers=verify_headers
                )
        except Exception as exc:
            return {"verification_status": "FAILURE",
                    "reason": f"Verification call failed ({type(exc).__name__})."}

        if resp.status_code != 200:
            return {"verification_status": "FAILURE",
                    "reason": f"PayPal verification API returned HTTP {resp.status_code}."}
        status = (resp.json() or {}).get("verification_status")
        if status != "SUCCESS":
            return {"verification_status": "FAILURE",
                    "reason": f"PayPal reported verification_status={status}."}
        return {"verification_status": "SUCCESS", "mode": "PAYPAL_API_VERIFIED"}
