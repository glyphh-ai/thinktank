"""Smoke tests for the storage bones: schema, relationships, vectors, infra."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.config import Settings
from app.db import init_db, make_engine, make_session_factory, session_scope
from app.models import (
    AnalysisRun,
    Document,
    Entity,
    EntityType,
    EvidenceRole,
    Fact,
    FactKind,
    Issue,
    IssueCategory,
    IssueEvidence,
    IssueStatus,
    PassageEmbedding,
    Project,
    RunKind,
    Severity,
    SourceType,
    Subscription,
    UsageEvent,
    UsageMetric,
    User,
    VerificationToken,
    sum_usage,
)
from app.models.user import TokenKind
from app.models.vectors import EMBEDDING_DIM, CortexLevel, HypervectorRecord
from app.plans import PlanTier, limits_for
from app.storage.blob import LocalDiskBlobStore
from app.storage.email import (
    ConsoleEmailSender,
    FileEmailSender,
    ResendEmailSender,
    get_email_sender,
    verification_pin_message,
)


@pytest.fixture
def factory(tmp_path):
    settings = Settings(database_url=f"sqlite+pysqlite:///{tmp_path}/test.db")
    engine = make_engine(settings)
    init_db(engine)
    return make_session_factory(engine)


def test_full_object_graph_persists(factory):
    with session_scope(factory) as s:
        user = User(email="writer@example.com", password_hash="hash")
        s.add(user)
        s.flush()

        s.add(Subscription(user_id=user.id, plan=PlanTier.PRO))
        project = Project(user_id=user.id, name="My Novel", word_count=120_000)
        s.add(project)
        s.flush()

        doc = Document(
            project_id=project.id,
            filename="novel.txt",
            source_type=SourceType.UPLOAD,
            blob_key=f"users/{user.id}/projects/{project.id}/documents/d1",
            word_count=120_000,
        )
        entity = Entity(project_id=project.id, canonical_name="Elena", type=EntityType.CHARACTER,
                        aliases=["the captain", "she"])
        s.add_all([doc, entity])
        s.flush()

        s.add(Fact(project_id=project.id, entity_id=entity.id, document_id=doc.id,
                   kind=FactKind.ATTRIBUTE, key="EYE_COLOR", value="GREEN", narrative_time=1))

        run = AnalysisRun(project_id=project.id, kind=RunKind.FIND, plan_at_run=PlanTier.PRO)
        s.add(run)
        s.flush()

        issue = Issue(run_id=run.id, project_id=project.id, category=IssueCategory.CONTINUITY,
                      severity=Severity.HIGH, title="Eye colour changes", description="green->blue",
                      confidence=0.9)
        s.add(issue)
        s.flush()
        s.add(IssueEvidence(issue_id=issue.id, document_id=doc.id, role=EvidenceRole.STATE_A,
                            char_start=10, char_end=40, quote="her green eyes"))

    # read back and verify relationships
    with session_scope(factory) as s:
        user = s.query(User).one()
        assert user.email == "writer@example.com"
        project = s.query(Project).one()
        assert project.documents[0].word_count == 120_000
        issue = s.query(Issue).one()
        assert issue.status is IssueStatus.OPEN  # default
        assert len(issue.evidence) == 1
        assert issue.run.kind is RunKind.FIND


def test_verification_token_and_cascade_delete(factory):
    with session_scope(factory) as s:
        user = User(email="a@b.com", password_hash="h")
        s.add(user)
        s.flush()
        s.add(VerificationToken(
            user_id=user.id, kind=TokenKind.EMAIL_PIN, secret_hash="pinhash",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        ))
        uid = user.id

    with session_scope(factory) as s:
        assert s.query(VerificationToken).count() == 1
        s.delete(s.get(User, uid))  # cascade removes tokens

    with session_scope(factory) as s:
        assert s.query(VerificationToken).count() == 0


def test_usage_ledger_sums(factory):
    with session_scope(factory) as s:
        user = User(email="u@v.com", password_hash="h")
        s.add(user)
        s.flush()
        s.add_all([
            UsageEvent(user_id=user.id, metric=UsageMetric.WORDS_ANALYZED, amount=1000),
            UsageEvent(user_id=user.id, metric=UsageMetric.WORDS_ANALYZED, amount=500),
            UsageEvent(user_id=user.id, metric=UsageMetric.FIXES_USED, amount=1),
        ])
        uid = user.id

    with session_scope(factory) as s:
        assert sum_usage(s, uid, UsageMetric.WORDS_ANALYZED) == 1500
        assert sum_usage(s, uid, UsageMetric.FIXES_USED) == 1


def test_embedding_roundtrips_through_sqlite(factory):
    vec = np.random.default_rng(0).standard_normal(EMBEDDING_DIM).astype(np.float32)
    with session_scope(factory) as s:
        user = User(email="e@m.com", password_hash="h")
        s.add(user)
        s.flush()
        project = Project(user_id=user.id, name="p")
        s.add(project)
        s.flush()
        s.add(PassageEmbedding(project_id=project.id, chunk_index=0, char_start=0,
                               char_end=10, text="hello", embedding=vec))

    with session_scope(factory) as s:
        got = s.query(PassageEmbedding).one().embedding
        assert got.shape == (EMBEDDING_DIM,)
        assert np.allclose(got, vec, atol=1e-6)


def test_hypervector_roundtrips_as_int8(factory):
    hv = (np.random.default_rng(1).integers(0, 2, 10_000) * 2 - 1).astype(np.int8)
    with session_scope(factory) as s:
        user = User(email="h@v.com", password_hash="h")
        s.add(user)
        s.flush()
        project = Project(user_id=user.id, name="p")
        s.add(project)
        s.flush()
        s.add(HypervectorRecord(project_id=project.id, level=CortexLevel.LAYER,
                                label="Elena", dim=10_000, data=hv))

    with session_scope(factory) as s:
        got = s.query(HypervectorRecord).one().data
        assert np.array_equal(got, hv)


def test_plan_limits():
    assert limits_for(PlanTier.FREE).can_fix is False
    assert limits_for(PlanTier.FREE).max_words_per_project == 3_000
    assert limits_for(PlanTier.PRO).can_fix is True
    assert limits_for(PlanTier.PRO).fixes_per_month == 50


def test_blob_store_roundtrip(tmp_path):
    store = LocalDiskBlobStore(str(tmp_path / "blobs"))
    key = "users/u1/projects/p1/documents/d1"
    store.put(key, b"manuscript bytes", "text/plain")
    assert store.exists(key)
    assert store.get(key) == b"manuscript bytes"
    store.delete(key)
    assert not store.exists(key)


def test_blob_store_rejects_traversal(tmp_path):
    store = LocalDiskBlobStore(str(tmp_path / "blobs"))
    with pytest.raises(ValueError):
        store.put("../escape", b"x")


def test_email_console_and_file_backends(tmp_path, capsys):
    settings = Settings(email_file_dir=str(tmp_path / "mail"))
    msg = verification_pin_message(settings, "user@example.com", "123456")

    ConsoleEmailSender().send(msg)
    assert "123456" in capsys.readouterr().out

    FileEmailSender(str(tmp_path / "mail")).send(msg)
    eml = list((tmp_path / "mail").glob("*.eml"))
    assert len(eml) == 1
    assert "123456" in eml[0].read_text()


def test_resend_backend_selection_and_payload():
    settings = Settings(email_backend="resend", resend_api_key="re_test_key")
    sender = get_email_sender(settings)
    assert isinstance(sender, ResendEmailSender)

    msg = verification_pin_message(settings, "user@example.com", "123456")
    payload = ResendEmailSender._payload(msg)
    assert payload["to"] == ["user@example.com"]
    assert payload["from"] == settings.email_from
    assert "123456" in payload["text"]


def test_resend_requires_api_key():
    with pytest.raises(ValueError):
        ResendEmailSender("")
