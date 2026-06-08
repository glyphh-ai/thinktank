"""Runtime configuration, read from the environment with dev-friendly defaults.

Nothing here requires Postgres, R2, or an SMTP server to be present: the defaults
run entirely on local SQLite, local-disk blobs, and console email so a developer
can register/verify/login/upload end to end with zero infrastructure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def load_env_file(path: str = ".env") -> None:
    """Minimal .env loader (no dependency). Existing env vars win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


load_env_file()


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- core ---
    env: str = field(default_factory=lambda: _env("APP_ENV", "dev"))
    # SQLite for dev; set DATABASE_URL to a postgresql+psycopg URL in prod.
    database_url: str = field(
        default_factory=lambda: _env("DATABASE_URL", "sqlite+pysqlite:///./var/app.db")
    )
    # Dev: sync additive schema changes (new tables/columns) on startup.
    # Disable in prod and use Alembic migrations instead.
    auto_migrate: bool = field(default_factory=lambda: _bool("AUTO_MIGRATE", True))

    # --- auth / tokens ---
    jwt_secret: str = field(default_factory=lambda: _env("JWT_SECRET", "dev-insecure-secret-change-me"))
    jwt_algorithm: str = field(default_factory=lambda: _env("JWT_ALGORITHM", "HS256"))
    access_token_ttl_seconds: int = field(default_factory=lambda: int(_env("ACCESS_TOKEN_TTL", "900")))
    refresh_token_ttl_seconds: int = field(
        default_factory=lambda: int(_env("REFRESH_TOKEN_TTL", str(60 * 60 * 24 * 30)))
    )
    email_pin_ttl_seconds: int = field(default_factory=lambda: int(_env("EMAIL_PIN_TTL", "600")))
    password_reset_ttl_seconds: int = field(
        default_factory=lambda: int(_env("PASSWORD_RESET_TTL", "3600"))
    )

    # --- blob storage (uploads) ---
    # "local" (dev, writes under blob_local_root) or "r2" (prod, S3-compatible).
    blob_backend: str = field(default_factory=lambda: _env("BLOB_BACKEND", "local"))
    blob_local_root: str = field(default_factory=lambda: _env("BLOB_LOCAL_ROOT", "./var/blobs"))
    r2_account_id: str = field(default_factory=lambda: _env("R2_ACCOUNT_ID", ""))
    r2_access_key_id: str = field(default_factory=lambda: _env("R2_ACCESS_KEY_ID", ""))
    r2_secret_access_key: str = field(default_factory=lambda: _env("R2_SECRET_ACCESS_KEY", ""))
    r2_bucket: str = field(default_factory=lambda: _env("R2_BUCKET", ""))

    # --- email ---
    # "console" (dev, prints), "file" (dev, writes .eml), "resend" (prod, HTTP
    # API), or "smtp" (prod / local Mailpit).
    email_backend: str = field(default_factory=lambda: _env("EMAIL_BACKEND", "console"))
    email_from: str = field(default_factory=lambda: _env("EMAIL_FROM", "noreply@plot-hole-finder.dev"))
    email_file_dir: str = field(default_factory=lambda: _env("EMAIL_FILE_DIR", "./var/mail"))
    resend_api_key: str = field(default_factory=lambda: _env("RESEND_API_KEY", ""))
    smtp_host: str = field(default_factory=lambda: _env("SMTP_HOST", "localhost"))
    smtp_port: int = field(default_factory=lambda: int(_env("SMTP_PORT", "1025")))
    smtp_user: str = field(default_factory=lambda: _env("SMTP_USER", ""))
    smtp_password: str = field(default_factory=lambda: _env("SMTP_PASSWORD", ""))
    smtp_use_tls: bool = field(default_factory=lambda: _bool("SMTP_USE_TLS", False))

    # --- extraction / verification LLM (Claude only; no fallback) ---
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-opus-4-8"))
    # Extraction chunk size. Opus has a 1M-token window, so use large chunks to
    # cut call count/latency (output tokens dominate cost regardless of chunking).
    extract_chunk_words: int = field(default_factory=lambda: int(_env("EXTRACT_CHUNK_WORDS", "15000")))
    extract_max_tokens: int = field(default_factory=lambda: int(_env("EXTRACT_MAX_TOKENS", "8000")))
    # Small passages for RAG evidence retrieval (independent of extraction chunks).
    passage_words: int = field(default_factory=lambda: int(_env("PASSAGE_WORDS", "400")))

    # --- billing (Stripe) ---
    # "disabled" (dev) or "stripe" (prod).
    billing_backend: str = field(default_factory=lambda: _env("BILLING_BACKEND", "disabled"))
    stripe_secret_key: str = field(default_factory=lambda: _env("STRIPE_SECRET_KEY", ""))
    stripe_webhook_secret: str = field(default_factory=lambda: _env("STRIPE_WEBHOOK_SECRET", ""))
    stripe_price_basic: str = field(default_factory=lambda: _env("STRIPE_PRICE_BASIC", ""))
    stripe_price_pro: str = field(default_factory=lambda: _env("STRIPE_PRICE_PRO", ""))

    # --- cache (Redis) ---
    # "memory" (dev, process-local) or "redis" (prod / shared).
    cache_backend: str = field(default_factory=lambda: _env("CACHE_BACKEND", "memory"))
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))

    # --- background jobs ---
    # "inline"   : API drains the queue in-request (tests / simple dev)
    # "thread"   : a background thread in the API process polls the queue (dev)
    # "external" : API only enqueues; run `python -m app.worker` separately (prod)
    job_runner: str = field(default_factory=lambda: _env("JOB_RUNNER", "inline"))
    job_lease_seconds: int = field(default_factory=lambda: int(_env("JOB_LEASE_SECONDS", "600")))
    job_max_attempts: int = field(default_factory=lambda: int(_env("JOB_MAX_ATTEMPTS", "3")))
    job_poll_interval: float = field(default_factory=lambda: float(_env("JOB_POLL_INTERVAL", "2.0")))

    # --- urls (links in emails, Stripe redirects, CORS) ---
    frontend_url: str = field(default_factory=lambda: _env("FRONTEND_URL", "http://localhost:3000"))
    cors_origins: str = field(default_factory=lambda: _env("CORS_ORIGINS", "http://localhost:3000"))

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")


def get_settings() -> Settings:
    return Settings()
