"""Runtime configuration, read from the environment with dev-friendly defaults.

Nothing here requires Postgres, R2, or an SMTP server to be present: the defaults
run entirely on local SQLite, local-disk blobs, and console email so a developer
can register/verify/login/upload end to end with zero infrastructure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


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
    # "console" (dev, prints), "file" (dev, writes .eml), or "smtp" (prod).
    email_backend: str = field(default_factory=lambda: _env("EMAIL_BACKEND", "console"))
    email_from: str = field(default_factory=lambda: _env("EMAIL_FROM", "noreply@plot-hole-finder.dev"))
    email_file_dir: str = field(default_factory=lambda: _env("EMAIL_FILE_DIR", "./var/mail"))
    smtp_host: str = field(default_factory=lambda: _env("SMTP_HOST", "localhost"))
    smtp_port: int = field(default_factory=lambda: int(_env("SMTP_PORT", "1025")))
    smtp_user: str = field(default_factory=lambda: _env("SMTP_USER", ""))
    smtp_password: str = field(default_factory=lambda: _env("SMTP_PASSWORD", ""))
    smtp_use_tls: bool = field(default_factory=lambda: _bool("SMTP_USE_TLS", False))

    # --- billing (Stripe) ---
    stripe_secret_key: str = field(default_factory=lambda: _env("STRIPE_SECRET_KEY", ""))
    stripe_webhook_secret: str = field(default_factory=lambda: _env("STRIPE_WEBHOOK_SECRET", ""))
    stripe_price_basic: str = field(default_factory=lambda: _env("STRIPE_PRICE_BASIC", ""))
    stripe_price_pro: str = field(default_factory=lambda: _env("STRIPE_PRICE_PRO", ""))

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")


def get_settings() -> Settings:
    return Settings()
