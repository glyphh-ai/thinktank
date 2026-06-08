"""The retrieval-backed memory: facts persist and drive cross-chunk detection."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.config import Settings
from app.db import init_db, make_engine, make_session_factory, session_scope
from app.models import Issue, MemoryFact, SourceType, User
from app.plans import PlanTier
from app.services import analysis as A
from app.services import projects as P
from app.services.billing import get_or_create_subscription
from app.storage.blob import get_blob_store
from fakes import StubEmbedder, StubExtractor, StubVerifier

# Eye-colour contradiction with the two halves far apart -> different chunks.
STORY = (
    "Petra's eyes were green when she arrived.\n\n"
    + "The hall was quiet and the day wore on. " * 400
    + "\n\nAt the end, Petra's eyes were blue under the lamps."
)


@pytest.fixture
def env(tmp_path):
    s = Settings(database_url=f"sqlite+pysqlite:///{tmp_path}/m.db",
                 blob_local_root=str(tmp_path / "blobs"), extract_chunk_words=300)
    eng = make_engine(s)
    init_db(eng)
    return s, make_session_factory(eng)


def test_memory_persists_and_detects_cross_chunk(env):
    settings, sf = env
    with session_scope(sf) as db:
        u = User(email="m@x", password_hash="h")
        db.add(u)
        db.flush()
        get_or_create_subscription(db, u.id).plan = PlanTier.PRO
        proj = P.create_project(db, get_blob_store(settings), u, "m", SourceType.PASTE, text=STORY)
        pid, uid = proj.id, u.id

    with session_scope(sf) as db:
        run = A.run_find(db, settings, get_blob_store(settings), StubEmbedder(),
                         StubExtractor(), StubVerifier(),
                         P.get_project(db, db.get(User, uid), pid), db.get(User, uid))
        assert run.stats["chunks"] > 1          # actually multiple chunks
        assert run.stats["facts"] >= 2          # facts were persisted to memory

    with session_scope(sf) as db:
        # memory backend populated, entity-indexed
        facts = db.scalars(select(MemoryFact).where(MemoryFact.entity == "Petra")).all()
        assert facts
        # contradiction found across chunks via retrieved memory
        issues = db.scalars(select(Issue).where(Issue.project_id == pid)).all()
        assert any("petra" in i.title.lower() for i in issues)
