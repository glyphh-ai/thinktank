"""
lattice.py — the one persistent lattice. Stop throwing runs away.

The store is SQLite with deterministic content-IDs and idempotent UPSERTs, so a
single file-backed lattice ACCUMULATES across runs: ingest more, it grows; ingest
the same thing twice, it's a no-op. This is the durable knowledge base every
other tool reads from and writes to.

  open_lattice()  -> the canonical Store at prime-lattice/lattice.db
  stats(store)    -> node counts by level/kind

The .db file is gitignored (binary, and rebuildable). The DURABLE record is the
committed source data in data/ + ingest.py, which reconstructs the lattice
deterministically. Within a session the .db persists and accumulates.
"""

from __future__ import annotations
import os
from store import Store

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lattice.db")


def open_lattice(path: str = DB_PATH) -> Store:
    return Store(path)


def stats(store: Store) -> dict:
    total = store.count()
    by_kind = dict(store.db.execute(
        "SELECT kind, count(*) FROM nodes GROUP BY kind ORDER BY count(*) DESC").fetchall())
    return {"total": total, "by_kind": by_kind}
