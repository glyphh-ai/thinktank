"""The analysis pipeline: rolling-memory detection over chunks.

Detection is Claude's: we read the document in chunks and let Claude maintain a
running memory of durable facts/constraints, flagging a chunk when it contradicts
memory with no event explaining it (``verifier.rolling_scan``). This is the
approach that won the eval bake-off; the earlier extraction+glyphh-cortex path
was both more complex and lower-recall, and has been removed. Passages are still
embedded so each issue can be grounded with evidence (jump-to-passage).

Scales past the context window because no single call sees the whole document.
For very large documents the running memory needs a retrieval backend (so it
isn't re-sent whole each chunk) -- see eval/README and DESIGN for that roadmap.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings
from ..embeddings import Embedder, search_passages
from ..errors import Forbidden
from ..llm import Verifier, WorldModelInconsistency
from ..models import (
    AnalysisRun,
    Issue,
    IssueCategory,
    IssueEvidence,
    MemoryFact,
    PassageEmbedding,
    Project,
    ProjectStatus,
    RunKind,
    RunStatus,
    Severity,
    UsageEvent,
    UsageMetric,
    User,
)
from ..models.narrative import EvidenceRole
from ..plans import limits_for
from ..text import chunk_text
from .billing import current_plan
from .projects import document_text


def run_find(
    session: Session,
    settings: Settings,
    blob_store,
    embedder: Embedder,
    extractor,            # accepted for call-site compatibility; detection is rolling_scan
    verifier: Verifier,
    project: Project,
    user: User,
    on_progress=None,
    **_ignored,           # legacy arm flags (recall/structured/holistic) -- no longer used
) -> AnalysisRun:
    def progress(step: str, fraction: float) -> None:
        if on_progress is not None:
            on_progress(step, fraction)

    plan = current_plan(session, user.id)
    run = AnalysisRun(project_id=project.id, kind=RunKind.FIND, status=RunStatus.RUNNING,
                      plan_at_run=plan, started_at=_now())
    session.add(run)
    project.status = ProjectStatus.ANALYZING
    session.flush()

    try:
        _reset_derived(session, project.id)
        progress("reading", 0.1)
        n_passages = _embed_passages(session, embedder, blob_store, project)
        progress("scanning", 0.4)

        # Retrieval-backed rolling memory: per chunk, retrieve only the facts for
        # entities mentioned in that chunk, scan, persist new facts. Memory is
        # never re-sent whole, so this scales past the context window.
        known_entities: set[str] = set()
        issues = n_chunks = n_facts = 0
        for text in _document_texts(blob_store, project):
            for chunk in chunk_text(text, target_words=settings.extract_chunk_words):
                n_chunks += 1
                lowered = chunk.text.lower()
                mentioned = [e for e in known_entities if e.lower() in lowered]
                known = _retrieve_facts(session, project.id, mentioned)
                result = verifier.scan_chunk(chunk.text, known)
                for f in result.new_facts:
                    session.add(MemoryFact(
                        project_id=project.id, entity=f["entity"][:255], key=f["key"][:128],
                        value=f["value"][:1024], kind=f.get("kind", "state"),
                        chunk_index=chunk.index))
                    known_entities.add(f["entity"])
                    n_facts += 1
                session.flush()
                for inc in result.issues:
                    if inc.entity:
                        _record_inc(session, run, project, embedder, inc)
                        issues += 1
        progress("grounding", 0.9)

        run.stats = {"passages": n_passages, "chunks": n_chunks,
                     "facts": n_facts, "issues": issues}
        run.status = RunStatus.SUCCEEDED
        run.finished_at = _now()
        project.status = ProjectStatus.READY
        session.add(UsageEvent(user_id=user.id, project_id=project.id,
                               metric=UsageMetric.WORDS_ANALYZED, amount=project.word_count))
        progress("done", 1.0)
    except Exception as exc:
        run.status = RunStatus.FAILED
        run.error = str(exc)
        run.finished_at = _now()
        project.status = ProjectStatus.FAILED
        raise
    return run


def run_fix(
    session: Session, settings: Settings, embedder: Embedder, verifier: Verifier,
    project: Project, user: User,
) -> AnalysisRun:
    limits = limits_for(current_plan(session, user.id))
    if not limits.can_fix:
        raise Forbidden("fix-this-for-me requires the Pro plan")

    run = AnalysisRun(project_id=project.id, kind=RunKind.FIX, status=RunStatus.RUNNING,
                      plan_at_run=limits.tier, started_at=_now())
    session.add(run)
    session.flush()

    open_issues = list(session.scalars(
        select(Issue).where(Issue.project_id == project.id, Issue.suggested_fix.is_(None))
    ))
    fixed = 0
    for issue in open_issues:
        evidence = [ev.quote for ev in issue.evidence]
        suggestion = verifier.suggest_fix(
            {"entity": issue.title, "key": issue.category.value, "description": issue.description},
            evidence,
        )
        issue.suggested_fix = suggestion.text
        fixed += 1

    session.add(UsageEvent(user_id=user.id, project_id=project.id,
                           metric=UsageMetric.FIXES_USED, amount=fixed))
    run.stats = {"fixes": fixed}
    run.status = RunStatus.SUCCEEDED
    run.finished_at = _now()
    return run


# --- steps -------------------------------------------------------------------

def _document_texts(blob_store, project) -> list[str]:
    return [document_text(blob_store, doc) for doc in project.documents]


def _retrieve_facts(session, project_id, entities, limit: int = 500) -> list[dict]:
    """Retrieve stored facts for the entities mentioned in the current chunk.
    Bounded, entity-indexed -> this is what keeps memory from being re-sent whole."""
    if not entities:
        return []
    rows = session.scalars(
        select(MemoryFact)
        .where(MemoryFact.project_id == project_id, MemoryFact.entity.in_(entities))
        .order_by(MemoryFact.chunk_index.desc())
        .limit(limit)
    ).all()
    return [{"entity": r.entity, "key": r.key, "value": r.value, "kind": r.kind} for r in rows]


def _embed_passages(session, embedder, blob_store, project) -> int:
    """Embed small passages for evidence retrieval (jump-to-passage)."""
    count = 0
    for doc in project.documents:
        chunks = chunk_text(document_text(blob_store, doc), target_words=400)
        if not chunks:
            continue
        vectors = embedder.embed([c.text for c in chunks])
        for c, vec in zip(chunks, vectors):
            session.add(PassageEmbedding(
                project_id=project.id, document_id=doc.id, chunk_index=c.index,
                char_start=c.char_start, char_end=c.char_end, text=c.text, embedding=vec,
            ))
            count += 1
    session.flush()
    return count


def _record_inc(session, run, project, embedder, inc: WorldModelInconsistency):
    """Record a detected inconsistency as an Issue, grounded with RAG evidence."""
    va, vb = inc.value_a, inc.value_b
    issue = Issue(
        run_id=run.id, project_id=project.id, category=_category(inc.category),
        severity=_severity(inc.severity),
        title=f"{inc.entity}: {inc.key} ({va} -> {vb})" if (va or vb) else f"{inc.entity}: {inc.key}",
        description=inc.explanation, confidence=inc.confidence,
    )
    session.add(issue)
    session.flush()
    for role, value in ((EvidenceRole.STATE_A, va), (EvidenceRole.STATE_B, vb)):
        neighbors = search_passages(session, project.id,
                                    embedder.embed_one(f"{inc.entity} {inc.key} {value}"), k=2)
        passage = neighbors[0].passage if neighbors else None
        session.add(IssueEvidence(
            issue_id=issue.id, document_id=passage.document_id if passage else None, role=role,
            char_start=passage.char_start if passage else 0,
            char_end=passage.char_end if passage else 0,
            quote=(passage.text[:240] if passage else f"{inc.entity} {inc.key}={value}"),
        ))
    return issue


def _reset_derived(session, project_id):
    """Clear prior derived data so re-runs are idempotent."""
    for model in (Issue, PassageEmbedding, MemoryFact):
        for row in session.scalars(select(model).where(model.project_id == project_id)):
            session.delete(row)
    session.flush()


def _now():
    return datetime.now(timezone.utc)


def _category(value: str) -> IssueCategory:
    try:
        return IssueCategory(value)
    except ValueError:
        return IssueCategory.CONTINUITY


def _severity(value: str) -> Severity:
    try:
        return Severity(value)
    except ValueError:
        return Severity.MEDIUM
