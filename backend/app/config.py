"""
Configuration settings for PayGuard AI backend.
Reads from environment variables and provides structured app settings.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "PayGuard AI"
    APP_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # PayPal Sandbox Configuration
    PAYPAL_CLIENT_ID: Optional[str] = None
    PAYPAL_CLIENT_SECRET: Optional[str] = None
    PAYPAL_MODE: str = "sandbox"
    PAYPAL_API_BASE: str = "https://api-m.sandbox.paypal.com"
    PAYPAL_WEBHOOK_ID: Optional[str] = None

    # Where the buyer's browser lands after PayPal approval (must reach this backend),
    # and where the backend then sends them (the Streamlit dashboard).
    PUBLIC_BASE_URL: str = "http://localhost:8000"
    DASHBOARD_URL: str = "http://localhost:8501"

    # LLM Investigation Agent Configuration
    GEMINI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gemini-3.5-flash"  # gemini-2.0/2.5-flash are no longer available to new keys

    # Risk Engine Decision Thresholds
    LOW_RISK_THRESHOLD: int = 30
    HIGH_RISK_THRESHOLD: int = 70

    # Step-up verification (simulated OTP for MEDIUM-risk orders)
    STEP_UP_OTP_TTL_SECONDS: int = 300
    STEP_UP_MAX_ATTEMPTS: int = 3      # wrong codes per challenge before escalation
    STEP_UP_MAX_CHALLENGES: int = 3    # codes issued per order before escalation
    STEP_UP_SECRET: Optional[str] = None  # HMAC key for stored code hashes; random per process if unset

    # API access (see core/security.py). Keys must be >= 24 characters; unset -> protected endpoints refuse.
    MERCHANT_API_KEY: Optional[str] = None
    ANALYST_API_KEYS: Optional[str] = None  # "analyst.id:key,analyst.id2:key2"

    # Demo Mode (also controls whether simulated OTP codes are shown on screen)
    DEMO_MODE: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
