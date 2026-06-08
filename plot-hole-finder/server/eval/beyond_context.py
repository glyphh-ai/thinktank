"""Beyond-the-context-window test: plot holes spanning >1M tokens.

Three baselines compared on the same >1M-token document:

1. single-pass     -- whole doc in one call. Cannot run (exceeds 1M tokens).
2. chunked-baseline-- Claude single-pass on each chunk, findings unioned. Runs,
                      but cannot see a plot hole whose halves are in different
                      chunks (i.e. the >1M-apart ones).
3. pipeline        -- chunk-extract -> compressed world model -> holistic review.
                      Carries facts across chunks, so it sees both halves.

    python -m eval.beyond_context   # real Claude; costs ~$12-15, takes minutes
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import tempfile
import time

import numpy as np
from sqlalchemy import select

from app.config import get_settings
from app.db import init_db, make_engine, make_session_factory, session_scope
from app.embeddings.base import Embedder, l2_normalize
from app.llm import get_extractor, get_verifier
from app.models import Document, Issue, Project, ProjectStatus, SourceType, User
from app.plans import PlanTier
from app.services import analysis as analysis_service
from app.services import projects as P
from app.services.billing import get_or_create_subscription
from app.storage.blob import LocalDiskBlobStore
from app.text import chunk_text, word_count
from .corpus_xl import build_xl_corpus
from .scoring import Benchmark, detail, score_case

PRICING = {"claude-opus-4-8": (5.0, 25.0), "claude-sonnet-4-6": (3.0, 15.0)}


class _Embedder(Embedder):  # evidence-only, not scored
    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                out[i, int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little") % self.dim] += 1
        return l2_normalize(out)


def main() -> None:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY required")
    in_m, out_m = PRICING.get(settings.anthropic_model, (5.0, 25.0))
    chunk_words = 40_000

    case = build_xl_corpus()
    wc = word_count(case.text)
    extractor, verifier = get_extractor(settings), get_verifier(settings)

    def spend() -> float:
        u, v = extractor.usage, verifier.usage
        return ((u["input_tokens"] + v["input_tokens"]) / 1e6 * in_m
                + (u["output_tokens"] + v["output_tokens"]) / 1e6 * out_m)

    try:
        tokens = extractor._client.messages.count_tokens(
            model=settings.anthropic_model,
            messages=[{"role": "user", "content": case.text}]).input_tokens
    except Exception:
        tokens = int(wc * 1.3)
    chunks = chunk_text(case.text, target_words=chunk_words)
    print(f"document: {wc:,} words = {tokens:,} tokens, {len(chunks)} chunks | "
          f"{len(case.plants)} plot holes + {len(case.decoys)} decoys\n")

    # 1) single pass over the whole doc -- can't ingest >1M tokens.
    print("[1] single-pass (whole document):")
    try:
        verifier.single_pass(case.text)
        print("    unexpectedly succeeded\n")
    except Exception as exc:
        print(f"    FAILED: {type(exc).__name__}: {str(exc)[:120]}\n")

    # 2) chunked baseline: single-pass per chunk, unioned.
    print("[2] chunked-baseline (Claude single-pass per chunk, unioned):")
    c0 = spend()
    t0 = time.time()
    cb_issues = []
    for ch in chunks:
        for inc in verifier.single_pass(ch.text):
            cb_issues.append({"title": f"{inc.entity}: {inc.key}",
                              "description": inc.explanation, "category": inc.category})
    cb = Benchmark([score_case(case, cb_issues)])
    print(f"    recall={cb.recall:.0%}  precision={cb.precision:.0%}  "
          f"decoy_violations={cb.decoy_violations} | {time.time()-t0:.0f}s | ${spend()-c0:.2f}\n")

    # 3) pipeline: chunk-extract -> world model -> holistic review.
    settings = dataclasses.replace(settings, auto_migrate=False,
                                   database_url=f"sqlite+pysqlite:///{tempfile.mkdtemp()}/xl.db",
                                   extract_chunk_words=chunk_words)
    engine = make_engine(settings)
    init_db(engine)
    sf = make_session_factory(engine)
    blob = LocalDiskBlobStore(tempfile.mkdtemp())
    with session_scope(sf) as s:
        user = User(email="xl@bench", password_hash="x")
        s.add(user)
        s.flush()
        get_or_create_subscription(s, user.id).plan = PlanTier.PRO
        project = Project(user_id=user.id, name="xl", status=ProjectStatus.CREATED, word_count=wc)
        s.add(project)
        s.flush()
        key = f"users/{user.id}/projects/{project.id}/d"
        blob.put(key, case.text.encode("utf-8"))
        s.add(Document(project_id=project.id, filename="xl.txt", source_type=SourceType.PASTE,
                       blob_key=key, text_blob_key=key, byte_size=len(case.text), word_count=wc))
        pid, uid = project.id, user.id

    print("[3] pipeline (chunk-extract -> world model -> holistic review):")
    c0 = spend()
    t0 = time.time()
    with session_scope(sf) as s:
        run = analysis_service.run_find(
            s, settings, blob, _Embedder(), extractor, verifier,
            P.get_project(s, s.get(User, uid), pid), s.get(User, uid),
            recall=False, structured=True, holistic=True)
        stats = run.stats
    with session_scope(sf) as s:
        pl_issues = [{"title": i.title, "description": i.description, "category": i.category.value}
                     for i in s.scalars(select(Issue).where(Issue.project_id == pid))]
    pl = Benchmark([score_case(case, pl_issues)])
    found, flagged = detail(case, pl_issues)
    print(f"    extract_chunks={stats['extract_chunks']} facts={stats['facts']} "
          f"issues={stats['issues']} | {time.time()-t0:.0f}s | ${spend()-c0:.2f}")
    print(f"    recall={pl.recall:.0%}  precision={pl.precision:.0%}  "
          f"decoy_violations={pl.decoy_violations}\n")
    print("    plot holes:", " ".join(("+" if ok else "-") + p.entity.split()[0] for p, ok in found))
    print("    decoys flagged:",
          " ".join(d.entity.split()[0] for d, bad in flagged if bad) or "(none)")


if __name__ == "__main__":
    main()
