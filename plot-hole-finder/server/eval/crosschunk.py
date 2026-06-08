"""Cross-chunk architecture bake-off (cheap).

Takes the in-context corpus but chunks it SMALL so every planted plot hole has
its two halves in different chunks -- the cross-chunk regime -- then compares
three approaches that all read the doc in chunks:

1. chunked-baseline -- Claude single-pass per chunk, unioned. No memory across
                       chunks, so it should miss cross-chunk holes.
2. rolling-memory   -- Claude maintains a running memory.md across chunks (the
                       Agent-SDK memory-tool approach). Catches a cross-chunk hole
                       only if it wrote the earlier fact to memory.
3. pipeline         -- chunk-extract -> world model -> holistic review.

Cheap (~$1-2): the whole doc is only ~16k words; we just slice it thin.

    python -m eval.crosschunk
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
from app.models import Issue, SourceType, User
from app.plans import PlanTier
from app.services import analysis as analysis_service
from app.services import projects as P
from app.services.billing import get_or_create_subscription
from app.storage.blob import LocalDiskBlobStore
from app.text import chunk_text
from .corpus import build_corpus
from .scoring import Benchmark, score_case

PRICING = {"claude-opus-4-8": (5.0, 25.0), "claude-sonnet-4-6": (3.0, 15.0)}
CHUNK_WORDS = 1500  # small, to force plot-hole halves into different chunks


class _Embedder(Embedder):
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

    case = build_corpus(target_words=16_000)
    extractor, verifier = get_extractor(settings), get_verifier(settings)
    chunks = chunk_text(case.text, target_words=CHUNK_WORDS)

    def spend():
        u, v = extractor.usage, verifier.usage
        return ((u["input_tokens"] + v["input_tokens"]) / 1e6 * in_m
                + (u["output_tokens"] + v["output_tokens"]) / 1e6 * out_m)

    print(f"corpus: {len(case.text.split()):,} words sliced into {len(chunks)} "
          f"chunks of ~{CHUNK_WORDS}w | {len(case.plants)} plot holes + "
          f"{len(case.decoys)} decoys (all halves cross-chunk)\n")
    print(f"{'arm':<18}{'recall':>8}{'prec':>7}{'decoyX':>8}{'cost$':>8}{'sec':>7}")

    def report(name, issues, c0, t0):
        b = Benchmark([score_case(case, issues)])
        print(f"{name:<18}{b.recall:>7.0%}{b.precision:>7.0%}{b.decoy_violations:>8}"
              f"{spend()-c0:>8.2f}{time.time()-t0:>7.0f}")

    # 1) chunked baseline
    c0, t0 = spend(), time.time()
    issues = [{"title": f"{i.entity}: {i.key}", "description": i.explanation, "category": i.category}
              for ch in chunks for i in verifier.single_pass(ch.text)]
    report("chunked_baseline", issues, c0, t0)

    # 2) rolling memory (memory.md across chunks)
    c0, t0 = spend(), time.time()
    incs = verifier.rolling_scan([ch.text for ch in chunks])
    issues = [{"title": f"{i.entity}: {i.key}", "description": i.explanation, "category": i.category}
              for i in incs]
    report("rolling_memory", issues, c0, t0)

    # 3) pipeline
    c0, t0 = spend(), time.time()
    s2 = dataclasses.replace(settings, auto_migrate=False, extract_chunk_words=CHUNK_WORDS,
                             database_url=f"sqlite+pysqlite:///{tempfile.mkdtemp()}/cc.db")
    engine = make_engine(s2)
    init_db(engine)
    sf = make_session_factory(engine)
    blob = LocalDiskBlobStore(tempfile.mkdtemp())
    with session_scope(sf) as s:
        u = User(email="cc@bench", password_hash="x")
        s.add(u)
        s.flush()
        get_or_create_subscription(s, u.id).plan = PlanTier.PRO
        proj = P.create_project(s, blob, u, "cc", SourceType.PASTE, text=case.text)
        pid, uid = proj.id, u.id
    with session_scope(sf) as s:
        analysis_service.run_find(s, s2, blob, _Embedder(), extractor, verifier,
                                  P.get_project(s, s.get(User, uid), pid), s.get(User, uid),
                                  recall=False, structured=True, holistic=True)
    with session_scope(sf) as s:
        issues = [{"title": i.title, "description": i.description, "category": i.category.value}
                  for i in s.scalars(select(Issue).where(Issue.project_id == pid))]
    report("pipeline", issues, c0, t0)


if __name__ == "__main__":
    main()
