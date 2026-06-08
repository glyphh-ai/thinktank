"""Release/deploy hook: ensure the database schema is current.

Run during the platform's release phase (Heroku `release:` in the Procfile, or a
Kubernetes init container / job) before new code serves traffic. Today this runs
create-tables + additive sync; swap to ``alembic upgrade head`` once Alembic
migrations are in place for prod.
"""

from __future__ import annotations

import logging

from .config import get_settings
from .db import auto_migrate, init_db, make_engine


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    engine = make_engine(settings)
    if settings.auto_migrate:
        auto_migrate(engine)
    else:
        init_db(engine)
    logging.getLogger("plothole.release").info("schema is up to date")


if __name__ == "__main__":
    main()
