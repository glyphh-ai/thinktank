"""Offline benchmark gate: the heuristic backend must catch its planted holes
without flagging licensed changes or the clean control."""

from __future__ import annotations

import dataclasses

from app.config import Settings
from eval.harness import run


def test_offline_benchmark_meets_bar():
    settings = Settings(llm_backend="heuristic", embed_backend="hash", billing_backend="disabled")
    bench = run(settings, verbose=False)

    # Every heuristic-detectable plant is found, and nothing is over-flagged.
    assert bench.recall == 1.0, [dataclasses.asdict(r) for r in bench.results]
    assert bench.decoy_violations == 0  # licensed hair-dye change not flagged
    assert bench.false_positives == 0   # clean control stays clean
    assert bench.precision == 1.0
