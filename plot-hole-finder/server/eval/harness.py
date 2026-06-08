"""Comparative benchmark: run the same labeled corpus through every approach and
score them head-to-head on recall / precision / F1 / cost / latency.

    python -m eval.harness            # needs ANTHROPIC_API_KEY (real Claude)

Arms:
- claude_1m   : hand the whole document to Claude in one pass ("find all
                inconsistencies"). The baseline.
- structured  : extract -> glyphh cortex -> verify (per-attribute, exhaustive).
- holistic    : extract -> one LLM pass over the compressed world model.
- hybrid      : structured + holistic, deduped (the product).

Detection is real Claude end to end. Embeddings here are evidence-only and not
scored, so the harness uses a tiny deterministic embedder to avoid a model
download; production uses FastEmbed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import tempfile
import time

import numpy as np
from sqlalchemy import select

from app.config import Settings, get_settings
from app.db import init_db, make_engine, make_session_factory, session_scope
from app.embeddings.base import Embedder, l2_normalize
from app.llm import get_extractor, get_verifier
from app.models import Issue, SourceType, User
from app.plans import PlanTier
from app.services import analysis as analysis_service
from app.services import projects as project_service
from app.services.billing import get_or_create_subscription
from app.storage.blob import LocalDiskBlobStore
from .corpus import build_corpus
from .scoring import Benchmark, score_case

PRICING = {"claude-opus-4-8": (5.0, 25.0), "claude-sonnet-4-6": (3.0, 15.0),
           "claude-haiku-4-5": (1.0, 5.0)}


class _BenchEmbedder(Embedder):
    """Evidence-only deterministic embedder for the benchmark (not scored)."""

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                out[i, int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little") % self.dim] += 1
        return l2_normalize(out)


def _usage(extractor, verifier) -> tuple[int, int]:
    eu, vu = extractor.usage, verifier.usage
    return (eu["input_tokens"] + vu["input_tokens"], eu["output_tokens"] + vu["output_tokens"])


def run(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    if not settings.anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY required for the benchmark")
    in_per_m, out_per_m = PRICING.get(settings.anthropic_model, (5.0, 25.0))

    case = build_corpus()
    tmp = tempfile.mkdtemp()
    settings = dataclasses.replace(settings, database_url=f"sqlite+pysqlite:///{tmp}/bench.db",
                                   auto_migrate=False)
    engine = make_engine(settings)
    init_db(engine)
    sf = make_session_factory(engine)
    blob = LocalDiskBlobStore(f"{tmp}/blobs")
    embedder = _BenchEmbedder()
    extractor, verifier = get_extractor(settings), get_verifier(settings)

    with session_scope(sf) as s:
        u = User(email="bench@local", password_hash="x")
        s.add(u)
        s.flush()
        get_or_create_subscription(s, u.id).plan = PlanTier.PRO
        uid = u.id

    print(f"corpus: {case.id} | {len(case.text.split()):,} words | "
          f"{len(case.plants)} plants + {len(case.decoys)} decoys\n")
    print(f"{'arm':<12}{'recall':>8}{'prec':>7}{'f1':>6}{'issues':>8}"
          f"{'decoyX':>8}{'FP':>5}{'cost$':>8}{'sec':>7}")

    rows = {}
    for arm in ("claude_1m", "structured", "holistic", "hybrid"):
        in0, out0 = _usage(extractor, verifier)
        t0 = time.time()
        issues = _run_arm(arm, sf, settings, blob, embedder, extractor, verifier, uid, case)
        secs = time.time() - t0
        din, dout = (a - b for a, b in zip(_usage(extractor, verifier), (in0, out0)))
        cost = din / 1e6 * in_per_m + dout / 1e6 * out_per_m
        bench = Benchmark([score_case(case, issues)])
        rows[arm] = bench
        print(f"{arm:<12}{bench.recall:>7.0%}{bench.precision:>7.0%}{bench.f1:>6.0%}"
              f"{len(issues):>8}{bench.decoy_violations:>8}{bench.false_positives:>5}"
              f"{cost:>8.3f}{secs:>7.0f}")
    return rows


def _run_arm(arm, sf, settings, blob, embedder, extractor, verifier, uid, case) -> list[dict]:
    if arm == "claude_1m":
        incs = verifier.single_pass(case.text)
        return [{"title": f"{i.entity}: {i.key}", "description": i.explanation,
                 "category": i.category} for i in incs]

    structured, holistic = {"structured": (True, False), "holistic": (False, True),
                            "hybrid": (True, True)}[arm]
    with session_scope(sf) as s:
        proj = project_service.create_project(s, blob, s.get(User, uid), arm,
                                              SourceType.PASTE, text=case.text)
        pid = proj.id
    with session_scope(sf) as s:
        analysis_service.run_find(s, settings, blob, embedder, extractor, verifier,
                                  project_service.get_project(s, s.get(User, uid), pid),
                                  s.get(User, uid), structured=structured, holistic=holistic)
    with session_scope(sf) as s:
        return [{"title": i.title, "description": i.description, "category": i.category.value}
                for i in s.scalars(select(Issue).where(Issue.project_id == pid))]


if __name__ == "__main__":
    run()
