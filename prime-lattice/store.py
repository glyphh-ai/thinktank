"""
store.py — schema-on-write storage for the primitive, built to scale.

Requirement: hundreds of millions of rows, written concurrently by a swarm.
That forces five decisions, all visible below:

  1. SCHEMA-ON-WRITE. Every record is validated at the door; a bad write is
     rejected, never persisted. The schema ENCODES THE ARCHITECTURE'S LAWS:
       - a prime may be untyped (fungible cell)
       - anything above the crystallisation line MUST be typed
       - every substrate edge must point at a row that already exists
       - provenance is mandatory (a 100-agent swarm must stay auditable)
     The primitive is what everything compounds on, so one bad prime would
     corrupt every program above it. Validate now, not at read time.

  2. DETERMINISTIC CONTENT-IDs. id = hash(level,kind,value). Two agents that
     independently derive the same prime produce the SAME id, so writes are
     idempotent (UPSERT / ON CONFLICT DO NOTHING). No auto-increment = no
     central coordination point = the swarm scales horizontally.

  3. NORMALISED EDGES. The lattice is a graph; substrate links live in their
     own `edges` table indexed BOTH ways, so "what is this made of" and "what
     uses this prime" stay O(degree) at 100M+ rows instead of scanning JSON.

  4. BATCHED INGEST. Rows go in via executemany inside one transaction with
     WAL; per-row inserts cannot hit 100M in reasonable time.

  5. PORTABLE DDL. SQLite here (zero-dep prototype). The same schema is
     Postgres-ready: switch the engine, add hash partitioning on id and
     declarative partitioning by `level` for the swarm's write throughput.

Run:  python store.py            # demo + small benchmark
      python store.py 2000000    # benchmark N synthetic rows
"""

from __future__ import annotations
import hashlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field

SCHEMA_VERSION = 1
LEVELS = ("prime", "program", "system")


class SchemaError(ValueError):
    """Raised when a write violates the schema. Schema-on-write = it never
    reaches disk."""


# ---------------------------------------------------------------------------
# The record. `substrate` is the list of child ids = downward lattice edges.
# ---------------------------------------------------------------------------
@dataclass
class Node:
    level: str
    kind: str
    value: str
    type: str | None = None
    substrate: list[str] = field(default_factory=list)
    pathways: dict = field(default_factory=dict)
    agent: str = ""                 # provenance: who wrote it
    model: str = ""                 # provenance: which LLM authored it (build time)
    strained: bool = False
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = node_id(self.level, self.kind, self.value)


def node_id(level: str, kind: str, value: str) -> str:
    """Deterministic, content-addressed id. Same prime -> same id, always."""
    h = hashlib.blake2b(f"{level}\x1f{kind}\x1f{value}".encode(), digest_size=12)
    return f"{level[0]}_{h.hexdigest()}"


# ---------------------------------------------------------------------------
# DDL. CHECK constraints catch the engine-level invariants; the cross-row
# rules (type-above-prime, referential integrity) are enforced in validate().
# ---------------------------------------------------------------------------
DDL = """
CREATE TABLE IF NOT EXISTS nodes (
  id             TEXT PRIMARY KEY,
  level          TEXT NOT NULL CHECK (level IN ('prime','program','system')),
  kind           TEXT NOT NULL,
  type           TEXT,
  value          TEXT,
  degree         INTEGER NOT NULL DEFAULT 0,
  pathways       TEXT NOT NULL DEFAULT '{}',
  agent          TEXT NOT NULL,
  ts             INTEGER NOT NULL,
  model          TEXT NOT NULL DEFAULT '',
  strained       INTEGER NOT NULL DEFAULT 0,
  schema_version INTEGER NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS edges (
  parent_id TEXT NOT NULL,
  child_id  TEXT NOT NULL,
  pathway   TEXT NOT NULL DEFAULT 'compose',
  ord       INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (parent_id, child_id, pathway)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_nodes_level ON nodes(level);
CREATE INDEX IF NOT EXISTS idx_nodes_kind  ON nodes(kind);
CREATE INDEX IF NOT EXISTS idx_nodes_type  ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_edges_child ON edges(child_id);
"""

PRAGMAS = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;
PRAGMA cache_size=-200000;
PRAGMA mmap_size=1073741824;
"""


class Store:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.executescript(PRAGMAS)
        self.db.executescript(DDL)

    # -- schema-on-write -----------------------------------------------------
    def validate(self, n: Node, known: set[str] | None = None) -> None:
        if n.level not in LEVELS:
            raise SchemaError(f"{n.id}: bad level {n.level!r}")
        if not n.kind:
            raise SchemaError(f"{n.id}: kind required")
        if not n.agent:
            raise SchemaError(f"{n.id}: provenance.agent required")
        if n.level == "prime":
            # a fungible cell: may be untyped, must be atomic
            if n.substrate:
                raise SchemaError(f"{n.id}: a prime has no substrate")
        else:
            # above the crystallisation line: MUST be typed and composed
            if not n.type:
                raise SchemaError(f"{n.id}: {n.level} must be typed (function)")
            if not n.substrate:
                raise SchemaError(f"{n.id}: {n.level} must have substrate")
            if known is not None:
                for cid in n.substrate:
                    if cid not in known:
                        raise SchemaError(
                            f"{n.id}: substrate {cid} does not exist (dangling edge)")

    # -- batched, idempotent ingest -----------------------------------------
    def put_many(self, nodes: list[Node]) -> int:
        ids_in_batch = {n.id for n in nodes}
        # referential integrity: a child must already exist OR arrive in this batch
        needed = {c for n in nodes for c in n.substrate} - ids_in_batch
        known = set(ids_in_batch)
        if needed:
            known |= self._existing(needed)
        for n in nodes:
            self.validate(n, known)

        ts = int(time.time())
        rows = [(n.id, n.level, n.kind, n.type, n.value, len(n.substrate),
                 json.dumps(n.pathways), n.agent, ts, n.model,
                 int(n.strained), SCHEMA_VERSION) for n in nodes]
        edges = [(n.id, c, "compose", i)
                 for n in nodes for i, c in enumerate(n.substrate)]

        with self.db:   # one transaction
            self.db.executemany(
                "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO NOTHING", rows)
            if edges:
                self.db.executemany(
                    "INSERT INTO edges VALUES (?,?,?,?) "
                    "ON CONFLICT(parent_id,child_id,pathway) DO NOTHING", edges)
        return len(rows)

    def put(self, n: Node) -> int:
        return self.put_many([n])

    # -- reads ---------------------------------------------------------------
    def _existing(self, ids: set[str]) -> set[str]:
        out, batch = set(), list(ids)
        for i in range(0, len(batch), 900):           # SQLite param limit
            chunk = batch[i:i + 900]
            q = f"SELECT id FROM nodes WHERE id IN ({','.join('?' * len(chunk))})"
            out |= {r[0] for r in self.db.execute(q, chunk)}
        return out

    def count(self) -> int:
        return self.db.execute("SELECT count(*) FROM nodes").fetchone()[0]

    def substrate_of(self, node_id: str) -> list[str]:
        return [r[0] for r in self.db.execute(
            "SELECT child_id FROM edges WHERE parent_id=? ORDER BY ord", (node_id,))]

    def users_of(self, child_id: str) -> list[str]:
        return [r[0] for r in self.db.execute(
            "SELECT parent_id FROM edges WHERE child_id=?", (child_id,))]


# ===========================================================================
# DEMO — show schema-on-write accepting valid lattice writes and REJECTING
# invalid ones, then benchmark throughput and project to 100M.
# ===========================================================================
def demo(store: Store):
    print("=" * 64)
    print("SCHEMA-ON-WRITE — valid lattice writes accepted")
    print("=" * 64)
    letters = [Node("prime", "letter", ch, agent="seed") for ch in "cat"]
    store.put_many(letters)
    cat = Node("program", "word", "cat", type="noun",
               substrate=[l.id for l in letters], agent="swarm-7", model="llm")
    store.put(cat)
    lexicon = Node("system", "lexicon", "core", type="lexicon",
                   substrate=[cat.id], agent="seed")
    store.put(lexicon)
    print(f"  wrote {store.count()} nodes: 3 letter-primes, 1 word, 1 system")
    print(f"  cat is made of : {store.substrate_of(cat.id)[:3]} (3 letters)")
    print(f"  'c' is used by : {store.users_of(letters[0].id)} (reverse edge)")

    print("\n" + "=" * 64)
    print("SCHEMA-ON-WRITE — invalid writes rejected at the door")
    print("=" * 64)
    bad = [
        Node("program", "word", "untyped", substrate=[cat.id], agent="x"),   # no type
        Node("program", "word", "dangling", type="noun",
             substrate=["p_doesnotexist"], agent="x"),                       # bad edge
        Node("prime", "letter", "z", type="noun", agent="x"),                # prime w/ type? ok
        Node("program", "word", "noprov", type="noun", substrate=[cat.id]),  # no agent
    ]
    for n in bad:
        try:
            store.put(n)
            print(f"  ACCEPTED {n.kind}:{n.value}")
        except SchemaError as e:
            print(f"  REJECTED -> {e}")


def benchmark(store: Store, n: int):
    print("\n" + "=" * 64)
    print(f"BENCHMARK — bulk ingest {n:,} synthetic word-nodes")
    print("=" * 64)
    letters = [Node("prime", "letter", ch, agent="seed")
               for ch in "abcdefghijklmnopqrstuvwxyz"]
    store.put_many(letters)
    sub = [letters[0].id, letters[1].id, letters[2].id]

    t0 = time.time()
    BATCH = 20_000
    buf = []
    for i in range(n):
        buf.append(Node("program", "word", f"w{i}", type="noun",
                        substrate=sub, agent=f"agent-{i % 100}", model="llm"))
        if len(buf) >= BATCH:
            store.put_many(buf); buf = []
    if buf:
        store.put_many(buf)
    dt = time.time() - t0

    rate = n / dt if dt else 0
    print(f"  inserted {store.count():,} nodes in {dt:.2f}s  ->  {rate:,.0f} rows/s")
    print(f"  (validated + edges + reverse index, single SQLite writer)")
    proj = 100_000_000 / rate if rate else 0
    print(f"  projected 100M rows: ~{proj/60:.1f} min on one writer")
    print( "  at scale: shard by id hash across writers / Postgres partitions")
    print( "  -> throughput scales ~linearly because ids need no coordination")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
    store = Store(":memory:")
    demo(store)
    benchmark(Store(":memory:"), n)


if __name__ == "__main__":
    main()
