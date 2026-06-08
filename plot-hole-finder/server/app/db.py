"""Database engine/session wiring.

SQLite for dev, Postgres for prod -- chosen by ``DATABASE_URL``. On Postgres we
ensure the ``vector`` extension exists so pgvector columns work from day one.
Dev uses ``init_db`` (create_all); prod should use Alembic migrations.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

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
    """Create extensions (PG) and all tables. Dev convenience; prod uses Alembic."""
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


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
