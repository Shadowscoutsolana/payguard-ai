"""
PayGuard AI - Core Security Module.

- API-key authentication (FastAPI dependencies) with constant-time comparison.
    MERCHANT_API_KEY   : the merchant's checkout backend (create, analyze, step-up, capture).
    ANALYST_API_KEYS   : "analyst.id:key,analyst.id2:key2"; the analyst's identity comes from
                         the key, never from the request body.
  Fail closed: if keys are not configured, protected endpoints refuse every request.
- PayPal webhook header extraction and certificate-URL allowlist (SSRF guard).
- PII helpers.

This is prototype-grade access control (shared static keys), not production identity management.
"""

import hashlib
import hmac
import logging
from typing import Dict, Optional
from urllib.parse import urlparse

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from ..config import settings

logger = logging.getLogger("payguard.security")

PAYPAL_ALLOWED_CERT_DOMAINS = [
    "api.sandbox.paypal.com",
    "api.paypal.com",
    "api-m.sandbox.paypal.com",
    "api-m.paypal.com",
]
PAYPAL_ALLOWED_API_BASES = {"https://api-m.sandbox.paypal.com", "https://api-m.paypal.com"}
MIN_KEY_LENGTH = 24

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _usable(key: Optional[str]) -> bool:
    return bool(key) and not key.startswith("your_") and len(key) >= MIN_KEY_LENGTH


def merchant_key() -> Optional[str]:
    key = settings.MERCHANT_API_KEY
    return key if _usable(key) else None


def analyst_keys() -> Dict[str, str]:
    """Parses ANALYST_API_KEYS -> {analyst_id: key}; ignores malformed or weak entries."""
    result: Dict[str, str] = {}
    for entry in (settings.ANALYST_API_KEYS or "").split(","):
        if ":" not in entry:
            continue
        analyst_id, key = (part.strip() for part in entry.split(":", 1))
        if analyst_id and _usable(key):
            result[analyst_id] = key
    return result


class SecurityManager:
    @staticmethod
    def constant_time_compare(val1: str, val2: str) -> bool:
        """Compares two strings in constant time to prevent timing attacks."""
        if not val1 or not val2:
            return False
        return hmac.compare_digest(val1.encode("utf-8"), val2.encode("utf-8"))

    @staticmethod
    def validate_paypal_cert_url(cert_url: str) -> bool:
        """SSRF guard: certificate URL must be https on an official PayPal API domain."""
        if not cert_url:
            return False
        try:
            parsed = urlparse(cert_url)
            if parsed.scheme != "https":
                return False
            return any(parsed.netloc.lower() == domain or parsed.netloc.lower().endswith("." + domain)
                       for domain in PAYPAL_ALLOWED_CERT_DOMAINS)
        except Exception:
            return False

    @staticmethod
    def extract_paypal_webhook_headers(raw_headers: Dict[str, str]) -> Dict[str, Optional[str]]:
        h = {k.lower(): v for k, v in raw_headers.items()}
        return {
            "auth_algo": h.get("paypal-auth-algo"),
            "cert_url": h.get("paypal-cert-url"),
            "transmission_id": h.get("paypal-transmission-id"),
            "transmission_sig": h.get("paypal-transmission-sig"),
            "transmission_time": h.get("paypal-transmission-time"),
        }

    @staticmethod
    def validate_webhook_header_presence(headers: Dict[str, Optional[str]]) -> bool:
        required = ["auth_algo", "cert_url", "transmission_id", "transmission_sig", "transmission_time"]
        return all(bool(headers.get(field)) for field in required)

    @staticmethod
    def hash_pii(value: str) -> str:
        if not value:
            return "anon_user"
        return "usr_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]

    @staticmethod
    def mask_wallet_address(address: str) -> str:
        if not address or len(address) < 10:
            return address or "unlinked"
        return f"{address[:6]}...{address[-4:]}"


security_manager = SecurityManager()


# ---------------------------------------------------------------- authentication
def identify(presented: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Maps a presented key to a principal. Every configured key is compared (no early exit),
    so response time does not reveal which key, if any, partially matched.
    """
    match: Optional[Dict[str, str]] = None
    m = merchant_key()
    if m and security_manager.constant_time_compare(presented or "", m):
        match = {"role": "merchant", "id": "merchant"}
    for analyst_id, key in analyst_keys().items():
        if security_manager.constant_time_compare(presented or "", key) and match is None:
            match = {"role": "analyst", "id": analyst_id}
    return match


def _authorize(request: Request, presented: Optional[str], allowed_roles: set) -> Dict[str, str]:
    if not merchant_key() and not analyst_keys():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="API keys are not configured on the server; access refused.")
    principal = identify(presented)
    if principal is None or principal["role"] not in allowed_roles:
        # Log the attempt, never the presented key.
        logger.warning("Auth failed: %s %s (needs %s, key %s)", request.method, request.url.path,
                       "/".join(sorted(allowed_roles)), "missing" if not presented else "rejected")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid API key.",
                            headers={"WWW-Authenticate": "ApiKey"})
    return principal


def require_merchant(request: Request, key: Optional[str] = Security(api_key_header)) -> Dict[str, str]:
    return _authorize(request, key, {"merchant"})


def require_analyst(request: Request, key: Optional[str] = Security(api_key_header)) -> Dict[str, str]:
    return _authorize(request, key, {"analyst"})


def require_merchant_or_analyst(request: Request, key: Optional[str] = Security(api_key_header)) -> Dict[str, str]:
    return _authorize(request, key, {"merchant", "analyst"})


# ---------------------------------------------------------------- startup checks
def startup_security_problems() -> list:
    """Configuration problems that must stop the server in production (warnings otherwise)."""
    problems = []
    if settings.PAYPAL_API_BASE.rstrip("/") not in PAYPAL_ALLOWED_API_BASES:
        problems.append("PAYPAL_API_BASE must be an official PayPal API base URL.")
    if not merchant_key():
        problems.append(f"MERCHANT_API_KEY is missing or shorter than {MIN_KEY_LENGTH} characters.")
    if not analyst_keys():
        problems.append("ANALYST_API_KEYS has no valid 'analyst.id:key' entries.")
    if settings.ENVIRONMENT.lower() == "production" and settings.DEMO_MODE:
        problems.append("DEMO_MODE must be false in production (it reveals simulated step-up codes).")
    return problems
