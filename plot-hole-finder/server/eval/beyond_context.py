"""Beyond-the-context-window test: a plot hole spanning >1M tokens.

Shows (1) the single-pass baseline cannot even ingest the document, and (2) the
chunk-extract -> compressed-world-model -> holistic-review pipeline detects the
plot hole whose halves are >1M tokens apart.

    python -m eval.beyond_context   # real Claude; costs ~$5-8, takes minutes
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
from app.services.billing import get_or_create_subscription
from app.storage.blob import LocalDiskBlobStore
from app.text import word_count
from .corpus_xl import build_xl_corpus
from .scoring import score_case

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
    in_per_m, out_per_m = PRICING.get(settings.anthropic_model, (5.0, 25.0))

    case = build_xl_corpus()
    wc = word_count(case.text)
    extractor, verifier = get_extractor(settings), get_verifier(settings)

    # Exact token count (utility endpoint).
    try:
        tokens = extractor._client.messages.count_tokens(
            model=settings.anthropic_model,
            messages=[{"role": "user", "content": case.text}]).input_tokens
    except Exception as exc:
        tokens = int(wc * 1.3)
        print(f"(count_tokens failed: {exc}; estimated)")
    print(f"document: {wc:,} words = {tokens:,} tokens "
          f"({'OVER' if tokens > 1_000_000 else 'under'} the 1M context window)\n")

    # 1) Baseline: try to hand the whole thing to Claude in one pass.
    print("[baseline] single Claude pass over the whole document:")
    try:
        verifier.single_pass(case.text)
        print("    unexpectedly succeeded")
    except Exception as exc:
        print(f"    FAILED — cannot ingest >1M tokens: {type(exc).__name__}: {str(exc)[:160]}\n")

    # 2) Pipeline: chunk-extract -> world model -> holistic review.
    settings = dataclasses.replace(settings, auto_migrate=False,
                                   database_url=f"sqlite+pysqlite:///{tempfile.mkdtemp()}/xl.db",
                                   extract_chunk_words=40_000)
    engine = make_engine(settings)
    init_db(engine)
    sf = make_session_factory(engine)
    blob = LocalDiskBlobStore(tempfile.mkdtemp())

    with session_scope(sf) as s:
        user = User(email="xl@bench", password_hash="x")
        s.add(user)
        s.flush()
        get_or_create_subscription(s, user.id).plan = PlanTier.PRO
        # Bypass the plan word-limit directly (this doc exceeds even the Pro cap).
        project = Project(user_id=user.id, name="xl", status=ProjectStatus.CREATED, word_count=wc)
        s.add(project)
        s.flush()
        key = f"users/{user.id}/projects/{project.id}/d"
        blob.put(key, case.text.encode("utf-8"))
        s.add(Document(project_id=project.id, filename="xl.txt", source_type=SourceType.PASTE,
                       blob_key=key, text_blob_key=key, byte_size=len(case.text), word_count=wc))
        pid, uid = project.id, user.id

    print("[pipeline] chunk-extract -> world model -> holistic review:")
    t0 = time.time()
    with session_scope(sf) as s:
        from app.services import projects as P
        run = analysis_service.run_find(
            s, settings, blob, _Embedder(), extractor, verifier,
            P.get_project(s, s.get(User, uid), pid), s.get(User, uid),
            recall=False, structured=True, holistic=True,
        )
        stats = run.stats
    secs = time.time() - t0

    with session_scope(sf) as s:
        issues = [{"title": i.title, "description": i.description, "category": i.category.value}
                  for i in s.scalars(select(Issue).where(Issue.project_id == pid))]
    bench = score_case(case, issues)
    cost = ((extractor.usage["input_tokens"] + verifier.usage["input_tokens"]) / 1e6 * in_per_m +
            (extractor.usage["output_tokens"] + verifier.usage["output_tokens"]) / 1e6 * out_per_m)

    print(f"    extract_chunks={stats['extract_chunks']} facts={stats['facts']} "
          f"issues={stats['issues']} | {secs:.0f}s | ${cost:.2f}")
    print(f"    plot hole detected: {'YES' if bench.found else 'NO'} "
          f"(recall {bench.found}/{bench.plants})")
    for i in issues:
        print(f"      - {i['title']}")


if __name__ == "__main__":
    main()
