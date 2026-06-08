"""Database engine/session wiring.

SQLite for dev, Postgres for prod -- chosen by ``DATABASE_URL``. On Postgres we
ensure the ``vector`` extension exists so pgvector columns work from day one.
Dev uses ``init_db`` (create_all); prod should use Alembic migrations.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import logging

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

log = logging.getLogger("plothole.db")

from .config import Settings, get_settings
from .models import Base


def make_engine(settings: Settings | None = None) -> Engine:
    settings = settings or get_settings()
    connect_args = {}
    if settings.database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        # Ensure the SQLite file's directory exists.
        path = settings.database_url.split("///")[-1]
        if path and path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    engine = create_engine(settings.database_url, connect_args=connect_args, future=True)

    if settings.database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _fk_pragma(dbapi_conn, _record):  # enforce FKs on SQLite
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    return engine


def init_db(engine: Engine) -> None:
    """Create extensions (PG), all tables, and the ANN index. Dev convenience;
    prod should use Alembic migrations."""
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_passage_embeddings_hnsw "
                "ON passage_embeddings USING hnsw (embedding vector_cosine_ops)"
            ))


def auto_migrate(engine: Engine) -> list[str]:
    """Dev convenience: sync *additive* schema changes on startup.

    Creates new tables/indexes (via init_db) and ALTERs in any new columns that
    models gained since the DB was created -- so restarting the dev server picks
    up model changes without a manual migration. It deliberately does NOT handle
    drops, renames, or type changes; use Alembic for those and for prod.
    """
    init_db(engine)
    applied = _sync_new_columns(engine)
    if applied:
        log.info("auto-migrate added columns: %s", ", ".join(applied))
    return applied


def _sync_new_columns(engine: Engine) -> list[str]:
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    applied: list[str] = []
    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue  # create_all already handled brand-new tables
        existing_cols = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing_cols:
                continue
            col_type = column.type.compile(dialect=engine.dialect)
            ddl = f'ALTER TABLE {table.name} ADD COLUMN {column.name} {col_type}'
            # Added columns are nullable in dev (existing rows have no value);
            # carry a scalar default through when one is declared.
            default = getattr(column, "default", None)
            if default is not None and getattr(default, "is_scalar", False):
                ddl += f" DEFAULT {_literal(default.arg)}"
            with engine.begin() as conn:
                conn.execute(text(ddl))
            applied.append(f"{table.name}.{column.name}")
    return applied


def _literal(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
