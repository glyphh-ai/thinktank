"""Dev auto-migrate: additive schema sync on startup."""

from __future__ import annotations

from sqlalchemy import inspect, text

from app.config import Settings
from app.db import auto_migrate, make_engine


def test_auto_migrate_adds_missing_column(tmp_path):
    engine = make_engine(Settings(database_url=f"sqlite+pysqlite:///{tmp_path}/m.db"))

    # Simulate an "old" DB: a users table missing the stripe_customer_id column.
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE users (id VARCHAR(32) PRIMARY KEY, email VARCHAR(320), "
            "password_hash VARCHAR(255))"
        ))

    before = {c["name"] for c in inspect(engine).get_columns("users")}
    assert "stripe_customer_id" not in before

    applied = auto_migrate(engine)

    after = {c["name"] for c in inspect(engine).get_columns("users")}
    assert "stripe_customer_id" in after  # column was added
    assert any("users.stripe_customer_id" == a for a in applied)


def test_auto_migrate_creates_new_tables(tmp_path):
    engine = make_engine(Settings(database_url=f"sqlite+pysqlite:///{tmp_path}/m2.db"))
    auto_migrate(engine)
    tables = set(inspect(engine).get_table_names())
    assert {"users", "projects", "jobs", "passage_embeddings"} <= tables
