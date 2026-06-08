"""Run the analysis pipeline over the labeled cases and report a benchmark.

    python -m eval.harness            # uses the configured backend (.env)
    LLM_BACKEND=heuristic python -m eval.harness   # offline

Adapters are built once and reused across cases, so when the Claude extractor is
active its cumulative token usage yields a real cost figure for the whole run.
"""

from __future__ import annotations

import dataclasses
import tempfile

from sqlalchemy import select

from app.config import Settings, get_settings
from app.db import init_db, make_engine, make_session_factory, session_scope
from app.embeddings import get_embedder
from app.llm import get_extractor, get_verifier
from app.models import Issue, SourceType, User
from app.plans import PlanTier
from app.services import analysis as analysis_service
from app.services import projects as project_service
from app.services.billing import get_or_create_subscription
from app.storage.blob import LocalDiskBlobStore
from .cases import CASES, Case
from .scoring import Benchmark, CaseResult, score_case

# Per-1M-token pricing (input, output) for cost reporting.
PRICING = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def _run_case(session_factory, settings, blob, embedder, extractor, verifier,
              user_id: str, case: Case) -> list[dict]:
    with session_scope(session_factory) as s:
        user = s.get(User, user_id)
        project = project_service.create_project(
            s, blob, user, case.id, SourceType.PASTE, text=case.text
        )
        pid = project.id
    with session_scope(session_factory) as s:
        project = project_service.get_project(s, s.get(User, user_id), pid)
        analysis_service.run_find(s, settings, blob, embedder, extractor, verifier,
                                  project, s.get(User, user_id))
    with session_scope(session_factory) as s:
        issues = s.scalars(select(Issue).where(Issue.project_id == pid)).all()
        return [{"title": i.title, "description": i.description, "category": i.category.value}
                for i in issues]


def run(settings: Settings | None = None, verbose: bool = True) -> Benchmark:
    settings = settings or get_settings()
    tmp = tempfile.mkdtemp()
    settings = dataclasses.replace(settings, database_url=f"sqlite+pysqlite:///{tmp}/eval.db",
                                   auto_migrate=False)
    engine = make_engine(settings)
    init_db(engine)
    sf = make_session_factory(engine)
    blob = LocalDiskBlobStore(f"{tmp}/blobs")
    embedder, extractor, verifier = get_embedder(settings), get_extractor(settings), get_verifier(settings)

    offline = settings.llm_backend == "heuristic"
    cases = [c for c in CASES if not (offline and c.llm_only)]

    with session_scope(sf) as s:
        user = User(email="eval@bench.local", password_hash="x")
        s.add(user)
        s.flush()
        get_or_create_subscription(s, user.id).plan = PlanTier.PRO  # allow large docs / fix
        user_id = user.id

    results: list[CaseResult] = []
    for case in cases:
        issues = _run_case(sf, settings, blob, embedder, extractor, verifier, user_id, case)
        results.append(score_case(case, issues))

    bench = Benchmark(results)
    if verbose:
        _report(settings, bench, extractor)
    return bench


def _report(settings, bench: Benchmark, extractor) -> None:
    print(f"\n=== plot-hole-finder benchmark ({settings.llm_backend}"
          f"{'/' + settings.anthropic_model if settings.llm_backend == 'claude' else ''}) ===")
    print(f"{'case':<26}{'plants':>7}{'found':>7}{'issues':>7}{'decoyX':>7}{'FP':>5}")
    for r in bench.results:
        print(f"{r.case_id:<26}{r.plants:>7}{r.found:>7}{r.issues:>7}"
              f"{r.decoy_violations:>7}{r.false_positives:>5}")
    print(f"\nrecall={bench.recall:.0%}  precision={bench.precision:.0%}  "
          f"f1={bench.f1:.0%}  decoy_violations={bench.decoy_violations}  "
          f"false_positives={bench.false_positives}")

    usage = getattr(extractor, "usage", None)
    if usage and usage["calls"]:
        ip, op = PRICING.get(settings.anthropic_model, (3.0, 15.0))
        cost = usage["input_tokens"] / 1e6 * ip + usage["output_tokens"] / 1e6 * op
        print(f"claude: {usage['calls']} calls | in {usage['input_tokens']:,} / "
              f"out {usage['output_tokens']:,} tok | cost ${cost:.4f}")


if __name__ == "__main__":
    run()
