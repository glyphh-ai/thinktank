"""Beyond-the-context-window test.

A ~277k-word document (≈360k tokens) -- far larger than any single LLM context
window -- with one contradiction planted at the very start and contradicted at
the very end. The pipeline never loads the whole document into one context: it
chunks, extracts per chunk, builds the glyphh cortex, and detects the
cross-document contradiction via triage + structured localization. This is the
core thesis of the project, exercised end to end (offline, deterministic).
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.db import init_db, make_engine, make_session_factory, session_scope
from app.embeddings import get_embedder
from app.llm import get_extractor, get_verifier
from app.models import Issue, SourceType, User
from app.plans import PlanTier
from app.services import analysis as analysis_service
from app.services import projects as project_service
from app.services.billing import get_or_create_subscription
from app.storage.blob import get_blob_store
from app.text import word_count

TARGET_WORDS = 250_000  # safely beyond a 200k-token context window


def _build_document() -> str:
    filler = (
        "The parties hereby agree to the terms set forth in this binding "
        "agreement and acknowledge the obligations described herein in good faith. "
    ) * 3
    paras = ["Senator Hale was 60 years old when he first took the oath of office."]
    # enough filler paragraphs to blow past the context window (count incrementally)
    per_para = word_count(filler)
    needed = (TARGET_WORDS // per_para) + 1
    paras.extend([filler] * needed)
    paras.append("By the closing statement, Senator Hale was 40 years old per the record.")
    return "\n\n".join(paras)


@pytest.fixture
def env(tmp_path):
    settings = Settings(
        database_url=f"sqlite+pysqlite:///{tmp_path}/big.db",
        blob_local_root=str(tmp_path / "blobs"),
        llm_backend="heuristic", embed_backend="hash",
    )
    engine = make_engine(settings)
    init_db(engine)
    return settings, make_session_factory(engine)


def test_finds_contradiction_beyond_context_window(env):
    settings, sf = env
    text = _build_document()
    assert word_count(text) > 200_000  # beyond a single context window

    with session_scope(sf) as db:
        user = User(email="legal@firm.com", password_hash="h")
        db.add(user)
        db.flush()
        get_or_create_subscription(db, user.id).plan = PlanTier.PRO  # allow large docs
        project = project_service.create_project(
            db, get_blob_store(settings), user, "Deposition", SourceType.PASTE, text=text
        )
        project_id, user_id = project.id, user.id

    with session_scope(sf) as db:
        project = db.get(type(project), project_id)
        user = db.get(User, user_id)
        run = analysis_service.run_find(
            db, settings, get_blob_store(settings), get_embedder(settings),
            get_extractor(settings), get_verifier(settings), project, user,
        )
        stats = run.stats

    # The document was split into many chunks (never one context) ...
    assert stats["chunks"] > 200
    # ... yet the start/end contradiction was found.
    assert stats["issues"] >= 1

    with session_scope(sf) as db:
        issues = db.scalars(select_issues(project_id)).all()
        assert any("hale" in i.title.lower() for i in issues)
        # evidence spans point to opposite ends of the document
        spans = [(e.char_start, e.char_end) for i in issues for e in i.evidence]
        assert max(s[0] for s in spans) - min(s[0] for s in spans) > 100_000


def select_issues(project_id: str):
    from sqlalchemy import select

    return select(Issue).where(Issue.project_id == project_id)
