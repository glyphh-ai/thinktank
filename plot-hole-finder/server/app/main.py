"""FastAPI application factory.

Builds infra singletons onto ``app.state`` from settings, registers routers, and
maps domain errors to HTTP responses. ``create_app`` accepts overridden settings
so tests can run against a temp SQLite DB with offline adapters.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .db import init_db, make_engine, make_session_factory
from .embeddings import get_embedder
from .errors import AppError
from .llm import get_extractor, get_verifier
from .routers import analysis, auth, billing, projects
from .services.billing import get_billing_provider
from .storage.blob import get_blob_store
from .storage.email import get_email_sender


def create_app(settings: Settings | None = None, *, init: bool = True) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="plot-hole-finder", version="0.1.0")

    # --- infra singletons on app.state ---
    engine = make_engine(settings)
    if init:
        init_db(engine)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = make_session_factory(engine)
    app.state.blob_store = get_blob_store(settings)
    app.state.email_sender = get_email_sender(settings)
    app.state.embedder = get_embedder(settings)
    app.state.extractor = get_extractor(settings)
    app.state.verifier = get_verifier(settings)
    app.state.billing = get_billing_provider(settings)

    # --- CORS for the Next.js frontend ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- domain error -> HTTP ---
    @app.exception_handler(AppError)
    async def _app_error_handler(_request: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code,
                            content={"error": exc.code, "message": exc.message})

    @app.get("/health", tags=["meta"])
    def health():
        return {"status": "ok", "env": settings.env}

    for r in (auth.router, projects.router, analysis.router, billing.router):
        app.include_router(r)

    return app


# Module-level app for `uvicorn app.main:app`.
app = create_app()
