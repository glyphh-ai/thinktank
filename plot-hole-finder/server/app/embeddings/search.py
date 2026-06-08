"""Dialect-aware nearest-neighbour search over passage embeddings.

On Postgres, uses pgvector's ``<=>`` cosine-distance operator (ANN-indexable).
On SQLite (dev), falls back to brute-force cosine in numpy. Same signature, so
callers never branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.vectors import PassageEmbedding


@dataclass
class Neighbor:
    passage: PassageEmbedding
    score: float  # cosine similarity in [-1, 1]


def search_passages(
    session: Session, project_id: str, query: np.ndarray, k: int = 5
) -> list[Neighbor]:
    query = np.asarray(query, dtype=np.float32)

    if session.bind.dialect.name == "postgresql":
        # pgvector: cosine distance = 1 - cosine similarity.
        distance = PassageEmbedding.embedding.cosine_distance(query)
        rows = session.execute(
            select(PassageEmbedding, distance.label("distance"))
            .where(PassageEmbedding.project_id == project_id)
            .order_by(distance)
            .limit(k)
        ).all()
        return [Neighbor(passage=row[0], score=1.0 - float(row[1])) for row in rows]

    # SQLite brute force.
    rows = session.scalars(
        select(PassageEmbedding).where(PassageEmbedding.project_id == project_id)
    ).all()
    if not rows:
        return []
    mat = np.vstack([r.embedding for r in rows]).astype(np.float32)
    qn = np.linalg.norm(query) or 1.0
    sims = (mat @ query) / (np.linalg.norm(mat, axis=1) * qn + 1e-9)
    order = np.argsort(sims)[::-1][:k]
    return [Neighbor(passage=rows[i], score=float(sims[i])) for i in order]
