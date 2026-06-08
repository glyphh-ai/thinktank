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
    *,
    structured: bool = True,
    holistic: bool = True,
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
        passages, extracted = _ingest(session, settings, blob_store, embedder, extractor, project)
        entities, facts = _persist_world_model(session, project, passages, extracted)
        project.status = ProjectStatus.ANALYZING
        session.flush()
        progress("analyzing", 0.5)

        cortex, codebook, value_clean = _build_cortex(session, project, entities, facts)
        candidates = _generate_candidates(cortex, codebook, value_clean, entities, facts)
        progress("verifying", 0.7)

        recorded: set[tuple[str, str]] = set()
        issues = structured_hits = 0

        # Arm 1 (structured/cortex): exhaustive per-(entity,attribute) candidates,
        # evidence-grounded and LLM-verified.
        if structured:
            for cand in candidates:
                evidence = _retrieve_evidence(session, embedder, project.id, cand)
                public = {k: cand[k] for k in ("entity", "key", "value_a", "value_b")}
                verdict = verifier.verify(public, [e["quote"] for e in evidence])
                if not verdict.confirmed:
                    continue
                _record_issue(session, run, project, cand, verdict)
                recorded.add((cand["entity"].upper(), cand["key"].upper()))
                structured_hits += 1
                issues += 1

        # Arm 2 (holistic): one LLM pass over the compressed world model to catch
        # cross-entity / semantic / timeline / knowledge issues the slots miss.
        global_hits = 0
        if holistic:
            world_model = _serialize_world_model(entities, facts)
            for inc in verifier.review_world_model(world_model):
                slot = (inc.entity.strip().upper(), inc.key.strip().upper())
                if not inc.entity or slot in recorded:
                    continue
                _record_global_issue(session, run, project, embedder, inc)
                recorded.add(slot)
                global_hits += 1
                issues += 1
        progress("verifying", 0.9)

        run.stats = {
            "passages": len(passages), "extract_chunks": len(extracted),
            "entities": len(entities), "facts": len(facts),
            "candidates": len(candidates), "issues": issues,
            "structured_hits": structured_hits, "global_hits": global_hits,
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

def _ingest(session, settings, blob_store, embedder, extractor, project):
    """Two chunkings: small passages (embeddings + narrative grid) and large
    chunks (extraction, exploiting the 1M context to cut call count)."""
    passages: list[tuple] = []   # (doc, Chunk) in document order -> narrative grid
    extracted: list[tuple] = []  # (doc, entities, facts-with-absolute-offsets)
    for doc in project.documents:
        text = document_text(blob_store, doc)

        small = chunk_text(text, target_words=settings.passage_words)
        if small:
            vectors = embedder.embed([c.text for c in small])
            for c, vec in zip(small, vectors):
                session.add(PassageEmbedding(
                    project_id=project.id, document_id=doc.id, chunk_index=c.index,
                    char_start=c.char_start, char_end=c.char_end, text=c.text, embedding=vec,
                ))
                passages.append((doc, c))

        for c in chunk_text(text, target_words=settings.extract_chunk_words):
            ex = extractor.extract(c.text, c.char_start)
            extracted.append((doc, ex.entities, _locate_facts(ex.facts, c)))
    session.flush()
    return passages, extracted


def _locate_facts(facts, chunk):
    """Resolve absolute char offsets for facts the LLM didn't position (offsets
    left at 0) by searching the chunk text for the value -- reliable, and far
    cheaper than asking the model for offsets."""
    haystack = chunk.text.lower()
    for f in facts:
        if f.char_end > f.char_start:
            continue  # heuristic extractor already provided absolute offsets
        pos = haystack.find(f.value.lower())
        if pos >= 0:
            f.char_start = chunk.char_start + pos
            f.char_end = f.char_start + len(f.value)
        else:
            f.char_start = f.char_end = chunk.char_start
    return facts


def _narrative_grid(passages):
    """Map a (doc_id, char offset) to a global passage index for ordering."""
    import bisect

    by_doc: dict[str, list[tuple[int, int]]] = {}
    for gidx, (doc, c) in enumerate(passages):
        by_doc.setdefault(doc.id, []).append((c.char_start, gidx))

    def position(doc_id: str, offset: int) -> int:
        entries = by_doc.get(doc_id)
        if not entries:
            return 0
        starts = [e[0] for e in entries]
        i = max(0, bisect.bisect_right(starts, offset) - 1)
        return entries[i][1]

    return position


def _persist_world_model(session, project, passages, extracted):
    """Upsert entities (by canonical name) and persist facts; narrative_time comes
    from the passage grid so it's fine-grained regardless of extraction chunk size."""
    position = _narrative_grid(passages)
    entities: dict[str, Entity] = {}
    facts: list[Fact] = []
    for doc, ents, fact_list in extracted:
        for e in ents:
            if e.name not in entities:
                ent = Entity(project_id=project.id, canonical_name=e.name,
                             type=_entity_type(e.type), aliases=e.aliases)
                session.add(ent)
                session.flush()
                entities[e.name] = ent
        for f in fact_list:
            ent = entities.get(f.entity)
            fact = Fact(
                project_id=project.id, entity_id=ent.id if ent else None, document_id=doc.id,
                kind=_fact_kind(f.kind),
                # Normalize so grouping is consistent across chunks and backends.
                key=f.key.strip().upper(), value=f.value.strip().upper(), polarity=f.polarity,
                narrative_time=position(doc.id, f.char_start),
                char_start=f.char_start, char_end=f.char_end, confidence=f.confidence,
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


def _serialize_world_model(entities, facts) -> str:
    """Compact text view of the extracted world model for the holistic LLM pass.
    Tiny relative to the prose, so it fits in context even for huge documents."""
    by_entity: dict[str, list] = {}
    for f in facts:
        by_entity.setdefault(f.entity_id, []).append(f)
    lines: list[str] = []
    for ent in entities:
        ent_facts = sorted(by_entity.get(ent.id, []), key=lambda f: f.narrative_time)
        if not ent_facts:
            continue
        lines.append(f"{ent.canonical_name} ({ent.type.value}):")
        for f in ent_facts:
            tag = "event" if f.kind == FactKind.EVENT else f.key
            lines.append(f"  t{f.narrative_time}: {tag}={f.value}")
    return "\n".join(lines)


def _record_global_issue(session, run, project, embedder, inc):
    """Record a holistic-pass inconsistency, grounding it with RAG evidence."""
    issue = Issue(
        run_id=run.id, project_id=project.id, category=_category(inc.category),
        severity=_severity(inc.severity), title=f"{inc.entity}: {inc.key} inconsistency",
        description=inc.explanation, confidence=inc.confidence,
    )
    session.add(issue)
    session.flush()
    query = embedder.embed_one(f"{inc.entity} {inc.key} {inc.value_a} {inc.value_b}")
    for role, value in ((EvidenceRole.STATE_A, inc.value_a), (EvidenceRole.STATE_B, inc.value_b)):
        neighbors = search_passages(session, project.id, query, k=2)
        passage = neighbors[0].passage if neighbors else None
        session.add(IssueEvidence(
            issue_id=issue.id, document_id=passage.document_id if passage else None,
            role=role,
            char_start=passage.char_start if passage else 0,
            char_end=passage.char_end if passage else 0,
            quote=(passage.text[:240] if passage else f"{inc.entity} {inc.key}={value}"),
        ))
    return issue


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
