"""FastAPI application factory.

Builds infra singletons onto ``app.state`` from settings, registers routers, and
maps domain errors to HTTP responses. ``create_app`` accepts overridden settings
so tests can run against a temp SQLite DB with offline adapters.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import threading

from .cache import get_cache
from .config import Settings, get_settings
from .db import auto_migrate, init_db, make_engine, make_session_factory
from .embeddings import get_embedder
from .errors import AppError
from .llm import get_extractor, get_verifier
from .routers import analysis, auth, billing, jobs, projects
from .services.billing import get_billing_provider
from .storage.blob import get_blob_store
from .storage.email import get_email_sender
from .worker import Worker


def create_app(
    settings: Settings | None = None,
    *,
    init: bool = True,
    blob_store=None,
    email_sender=None,
    embedder=None,
    extractor=None,
    verifier=None,
) -> FastAPI:
    """Build the app. Adapters can be injected (tests pass stubs); otherwise the
    real Claude/FastEmbed/etc. adapters are constructed from settings."""
    settings = settings or get_settings()
    app = FastAPI(title="plot-hole-finder", version="0.1.0")

    # --- infra singletons on app.state (injected or built from settings) ---
    engine = make_engine(settings)
    if init:
        auto_migrate(engine) if settings.auto_migrate else init_db(engine)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = make_session_factory(engine)
    app.state.blob_store = blob_store or get_blob_store(settings)
    app.state.email_sender = email_sender or get_email_sender(settings)
    app.state.embedder = embedder or get_embedder(settings)
    app.state.extractor = extractor or get_extractor(settings)
    app.state.verifier = verifier or get_verifier(settings)
    app.state.billing = get_billing_provider(settings)
    app.state.cache = get_cache(settings)
    app.state.worker = Worker(
        app.state.session_factory, settings, blob_store=app.state.blob_store,
        embedder=app.state.embedder, extractor=app.state.extractor, verifier=app.state.verifier,
    )

    # In "thread" mode a daemon worker polls the queue inside the API process
    # (dev convenience). Prod uses "external" + `python -m app.worker`.
    if settings.job_runner == "thread":
        t = threading.Thread(target=app.state.worker.run_forever, daemon=True)
        t.start()
        app.state.worker_thread = t

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

    for r in (auth.router, projects.router, analysis.router, jobs.router, billing.router):
        app.include_router(r)

    return app


def __getattr__(name: str):
    # Lazily build the app for `uvicorn app.main:app` so merely importing this
    # module (e.g. for create_app in tests) doesn't construct real adapters.
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
