"""The analysis pipeline: extract -> index (facts + cortex + embeddings)
-> triage (glyphh) -> verify (LLM) -> issues.

Find detects inconsistencies; Fix (Pro) proposes edits for confirmed issues.
Designed to run as a background task; all heavy deps are injected so it runs
fully offline in dev with the heuristic extractor + hashing embedder.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings
from ..embeddings import Embedder, search_passages
from ..errors import Forbidden, NotFound
from ..glyphh_bridge import glyphh as g
from ..llm import Extractor, Verifier
from ..models import (
    AnalysisRun,
    Document,
    Entity,
    EntityType,
    Fact,
    FactKind,
    Issue,
    IssueCategory,
    IssueEvidence,
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
from ..models.vectors import CortexLevel, HypervectorRecord
from ..plans import limits_for
from ..text import chunk_text
from .billing import current_plan
from .projects import document_text

HV_DIM = 10_000
# Event keys that legitimately license a change to a given attribute.
_LICENSING = {
    "EYE_COLOR": {"WEAR_CONTACTS"},
    "HAIR": {"DYE_HAIR"},
}


def run_find(
    session: Session,
    settings: Settings,
    blob_store,
    embedder: Embedder,
    extractor: Extractor,
    verifier: Verifier,
    project: Project,
    user: User,
    on_progress=None,
) -> AnalysisRun:
    def progress(step: str, fraction: float) -> None:
        if on_progress is not None:
            on_progress(step, fraction)

    plan = current_plan(session, user.id)
    run = AnalysisRun(project_id=project.id, kind=RunKind.FIND, status=RunStatus.RUNNING,
                      plan_at_run=plan, started_at=_now())
    session.add(run)
    project.status = ProjectStatus.EXTRACTING
    session.flush()

    try:
        _reset_derived(session, project.id)
        progress("extracting", 0.15)
        chunks = _ingest(session, blob_store, embedder, extractor, project)
        entities, facts = _persist_world_model(session, project, chunks)
        project.status = ProjectStatus.ANALYZING
        session.flush()
        progress("analyzing", 0.5)

        cortex, codebook, value_clean = _build_cortex(session, project, entities, facts)
        candidates = _generate_candidates(cortex, codebook, value_clean, entities, facts)
        progress("verifying", 0.7)

        issues = 0
        for cand in candidates:
            evidence = _retrieve_evidence(session, embedder, project.id, cand)
            verdict = verifier.verify(cand, [e["quote"] for e in evidence])
            if not verdict.confirmed:
                continue
            _record_issue(session, run, project, cand, verdict)
            issues += 1

        run.stats = {
            "chunks": len(chunks), "entities": len(entities), "facts": len(facts),
            "candidates": len(candidates), "issues": issues,
        }
        run.status = RunStatus.SUCCEEDED
        run.finished_at = _now()
        project.status = ProjectStatus.READY
        session.add(UsageEvent(user_id=user.id, project_id=project.id,
                               metric=UsageMetric.WORDS_ANALYZED, amount=project.word_count))
        progress("done", 1.0)
    except Exception as exc:  # mark the run failed, surface the message
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


# --- pipeline steps ----------------------------------------------------------

def _ingest(session, blob_store, embedder, extractor, project) -> list:
    """Chunk every document, embed + persist passages, extract per chunk."""
    enriched = []
    for doc in project.documents:
        text = document_text(blob_store, doc)
        chunks = chunk_text(text)
        if not chunks:
            continue
        vectors = embedder.embed([c.text for c in chunks])
        for c, vec in zip(chunks, vectors):
            session.add(PassageEmbedding(
                project_id=project.id, document_id=doc.id, chunk_index=c.index,
                char_start=c.char_start, char_end=c.char_end, text=c.text, embedding=vec,
            ))
            extraction = extractor.extract(c.text, c.char_start)
            enriched.append((doc, c, extraction))
    session.flush()
    return enriched


def _persist_world_model(session, project, enriched):
    """Upsert entities (by canonical name) and persist facts with chunk time."""
    entities: dict[str, Entity] = {}
    facts: list[Fact] = []
    for doc, chunk, extraction in enriched:
        for e in extraction.entities:
            if e.name not in entities:
                ent = Entity(project_id=project.id, canonical_name=e.name,
                             type=_entity_type(e.type), aliases=e.aliases)
                session.add(ent)
                session.flush()
                entities[e.name] = ent
        for f in extraction.facts:
            ent = entities.get(f.entity)
            fact = Fact(
                project_id=project.id, entity_id=ent.id if ent else None, document_id=doc.id,
                kind=_fact_kind(f.kind),
                # Normalize so grouping is consistent across chunks and backends
                # (Claude returns mixed case; the heuristic returns upper).
                key=f.key.strip().upper(), value=f.value.strip().upper(), polarity=f.polarity,
                narrative_time=chunk.index, char_start=f.char_start, char_end=f.char_end,
                confidence=f.confidence,
            )
            session.add(fact)
            facts.append(fact)
    session.flush()
    return list(entities.values()), facts


def _build_cortex(session, project, entities, facts):
    """Build (and persist) the glyphh cortex: roles -> segments -> layers."""
    codebook = g.Codebook(HV_DIM, seed=1)
    values_by_key: dict[str, set[str]] = {}
    for f in facts:
        if f.kind == FactKind.ATTRIBUTE:
            values_by_key.setdefault(f.key, set()).add(f.value)
    value_clean = {
        key: g.value_cleanup(codebook, sorted(vals)) for key, vals in values_by_key.items()
    }

    cortex = g.Cortex(project.id, HV_DIM)
    by_entity = {e.id: e for e in entities}
    for ent in entities:
        layer = g.Layer(ent.canonical_name, HV_DIM)
        # group this entity's attribute facts into per-chunk segments
        segs: dict[int, "g.Segment"] = {}
        for f in facts:
            if f.entity_id != ent.id or f.kind != FactKind.ATTRIBUTE:
                continue
            seg = segs.get(f.narrative_time)
            if seg is None:
                seg = g.Segment(f"{ent.canonical_name}@{f.narrative_time}", HV_DIM,
                                narrative_time=f.narrative_time)
                segs[f.narrative_time] = seg
            seg.add(g.make_role(codebook, f.key, f.value))
        for seg in segs.values():
            layer.add(seg)
        if layer.segments:
            cortex.add(layer)
            session.add(HypervectorRecord(project_id=project.id, level=CortexLevel.LAYER,
                                          label=ent.canonical_name, dim=HV_DIM, data=layer.vector))
    session.add(HypervectorRecord(project_id=project.id, level=CortexLevel.CORTEX,
                                  label=project.id, dim=HV_DIM, data=cortex.vector))
    session.flush()
    return cortex, codebook, value_clean


def _generate_candidates(cortex, codebook, value_clean, entities, facts):
    """glyphh triage (contested attrs) + structured localization + licensing."""
    by_name = {e.canonical_name: e for e in entities}
    candidates = []
    for name, layer in cortex.layers.items():
        ent = by_name[name]
        for key, vclean in value_clean.items():
            contested = layer.contested(codebook, key, vclean)
            if len(contested) <= 1:
                continue  # cheap signal says this slot is consistent
            groups = _localize(ent, key, facts)
            if len(groups) <= 1:
                continue
            ordered = sorted(groups.items(), key=lambda kv: kv[1][0].narrative_time)
            (val_a, side_a), (val_b, side_b) = ordered[0], ordered[-1]
            if _has_licensing_event(ent, key, side_a[0].narrative_time, side_b[0].narrative_time, facts):
                continue  # legitimate change -- an event explains it
            candidates.append({
                "entity": name, "key": key, "value_a": val_a, "value_b": val_b,
                "fact_a": side_a[0], "fact_b": side_b[-1],
            })
    return candidates


def _localize(entity, key, facts):
    groups: dict[str, list[Fact]] = {}
    for f in facts:
        if f.entity_id == entity.id and f.kind == FactKind.ATTRIBUTE and f.key == key:
            groups.setdefault(f.value, []).append(f)
    return groups


def _has_licensing_event(entity, key, t_a, t_b, facts) -> bool:
    licensing = _LICENSING.get(key, set())
    lo, hi = sorted((t_a, t_b))
    for f in facts:
        if (f.entity_id == entity.id and f.kind == FactKind.EVENT and f.key in licensing
                and lo <= f.narrative_time <= hi):
            return True
    return False


def _retrieve_evidence(session, embedder, project_id, cand):
    """Pull source passages for the two sides (RAG evidence layer)."""
    evidence = []
    for fact in (cand["fact_a"], cand["fact_b"]):
        passage = session.scalar(
            select(PassageEmbedding).where(
                PassageEmbedding.project_id == project_id,
                PassageEmbedding.char_start <= fact.char_start,
                PassageEmbedding.char_end >= fact.char_end,
            )
        )
        quote = passage.text if passage else f"{cand['entity']} {cand['key']}={fact.value}"
        evidence.append({"quote": quote, "fact": fact})
    return evidence


def _record_issue(session, run, project, cand, verdict):
    issue = Issue(
        run_id=run.id, project_id=project.id, category=_category_for(cand["key"]),
        severity=Severity(verdict.severity), title=verdict.title or cand["entity"],
        description=verdict.explanation, confidence=verdict.confidence,
    )
    session.add(issue)
    session.flush()
    for role, fact in ((EvidenceRole.STATE_A, cand["fact_a"]), (EvidenceRole.STATE_B, cand["fact_b"])):
        session.add(IssueEvidence(
            issue_id=issue.id, document_id=fact.document_id, role=role,
            char_start=fact.char_start, char_end=fact.char_end,
            quote=f"{cand['entity']} {cand['key']}={fact.value}",
        ))
    return issue


# --- helpers -----------------------------------------------------------------

def _reset_derived(session, project_id):
    """Clear prior derived data so re-runs are idempotent."""
    for model in (Fact, Entity, PassageEmbedding, HypervectorRecord):
        for row in session.scalars(select(model).where(model.project_id == project_id)):
            session.delete(row)
    session.flush()


def _now():
    return datetime.now(timezone.utc)


def _entity_type(t: str) -> EntityType:
    try:
        return EntityType(t)
    except ValueError:
        return EntityType.CONCEPT


def _fact_kind(k: str) -> FactKind:
    try:
        return FactKind(k)
    except ValueError:
        return FactKind.ATTRIBUTE


def _category_for(key: str) -> IssueCategory:
    return {
        "AGE": IssueCategory.TIMELINE,
    }.get(key, IssueCategory.CONTINUITY)
