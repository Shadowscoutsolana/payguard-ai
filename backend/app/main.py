"""
PayGuard AI - Main FastAPI Application.
Synchronous pre-capture risk gate for PayPal Sandbox transactions with Ethereum/PYUSD graph intelligence.
"""

import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .core.security import startup_security_problems
from .api.payments import router as payments_router
from .api.fraud import router as fraud_router
from .api.webhooks import router as webhooks_router

logger = logging.getLogger("payguard")
IS_PRODUCTION = settings.ENVIRONMENT.lower() == "production"

# Refuse to start in production with an unsafe configuration; warn loudly otherwise.
_problems = startup_security_problems()
if _problems and IS_PRODUCTION:
    raise RuntimeError("Unsafe production configuration: " + " | ".join(_problems))
for _p in _problems:
    logger.warning("Security configuration: %s", _p)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Pre-capture risk analysis gate for PayPal transactions with PYUSD blockchain cluster intelligence.",
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
)

# CORS: only the dashboard origin, no cookies/credentials (auth is an explicit X-API-Key header).
# The Streamlit dashboard calls the API from its Python server, so browsers need no wider access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.DASHBOARD_URL.rstrip("/")],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    """Never return stack traces or internal messages to clients; log them server-side."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    """
    Measures and attaches server execution latency to every response.
    Validates adherence to the sub-100ms synchronous decision target SLA.
    """
    start_time = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Decision-Latency-Ms"] = f"{elapsed_ms:.2f}"
    return response


# Include API Routers
app.include_router(payments_router)
app.include_router(fraud_router)
app.include_router(webhooks_router)


@app.get("/")
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "ONLINE",
        "mode": "Sandbox Pre-Capture Risk Gate",
        "demo_mode": settings.DEMO_MODE,
        "paypal_configured": bool(settings.PAYPAL_CLIENT_ID and "your_" not in settings.PAYPAL_CLIENT_ID),
        "target_sla": "<100ms synchronous decision",
    }


@app.get("/health")
async def health_check():
    return {"status": "HEALTHY", "timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
