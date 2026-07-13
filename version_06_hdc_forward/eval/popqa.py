"""
popqa.py — real KBQA data for the grounding evaluation (PopQA, Mallen et al. 2023).
=================================================================================
PopQA = popularity-stratified entity questions over Wikidata triples. Popularity
(subject Wikipedia pageviews) is the difficulty axis: rare entities are where a
model's parametric knowledge fails and grounding matters.

We sample stratified by popularity, split BY ENTITY into:
  KB       — triples we store (in-KB queries: the fact IS grounded)
  HELDOUT  — real triples we do NOT store (out-of-KB queries: a grounded system
             must ABSTAIN; the fact is real, so the model may still know it)
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from typing import List, Tuple

COLS = ["id", "subj", "prop", "obj", "subj_id", "prop_id", "obj_id", "s_aliases",
        "o_aliases", "s_uri", "o_uri", "s_wiki_title", "o_wiki_title", "s_pop",
        "o_pop", "question", "possible_answers"]


@dataclass
class Item:
    subj: str
    prop: str
    obj: str
    question: str
    answers: List[str]
    pop: int
    tercile: str = "?"


def load(path: str) -> List[Item]:
    out: List[Item] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter="\t", fieldnames=COLS)
        next(r)  # header
        for row in r:
            try:
                answers = json.loads(row["possible_answers"])
            except Exception:
                answers = []
            answers = [a for a in ([row["obj"]] + answers) if a]
            try:
                pop = int(row["s_pop"])
            except Exception:
                pop = 0
            if row["subj"] and row["prop"] and row["obj"] and row["question"]:
                out.append(Item(row["subj"], row["prop"], row["obj"], row["question"], answers, pop))
    return out


def _bucket(sorted_pops, p):
    n = len(sorted_pops)
    if p <= sorted_pops[n // 3]:
        return "rare"
    if p <= sorted_pops[2 * n // 3]:
        return "mid"
    return "popular"


def sample(items: List[Item], per_tercile: int = 80, seed: int = 0) -> List[Item]:
    """Stratified sample across popularity terciles (balanced difficulty)."""
    import random
    rng = random.Random(seed)
    pops = sorted(i.pop for i in items)
    for it in items:
        it.tercile = _bucket(pops, it.pop)
    by = {"rare": [], "mid": [], "popular": []}
    for it in items:
        by[it.tercile].append(it)
    picked: List[Item] = []
    for t, group in by.items():
        rng.shuffle(group)
        picked += group[:per_tercile]
    rng.shuffle(picked)
    return picked


def split_by_entity(items: List[Item], kb_frac: float = 0.5) -> Tuple[List[Item], List[Item]]:
    """Split so HELDOUT entities are entirely absent from the KB (clean abstention)."""
    kb, held = [], []
    for it in items:
        h = int(hashlib.sha256(it.subj.encode()).hexdigest(), 16) % 1000
        (kb if h < kb_frac * 1000 else held).append(it)
    return kb, held


def render_passage(it: Item) -> str:
    return f"The {it.prop} of {it.subj} is {it.obj}."


def matches(pred: str, answers: List[str]) -> bool:
    """PopQA metric: prediction counts correct if it contains any gold alias."""
    p = pred.lower()
    return any(a.lower() in p for a in answers if a.strip())


if __name__ == "__main__":
    import os
    SP = os.environ.get("SCRATCH", "/private/tmp/claude-501/-Users-timmetim-development-yo/"
                        "88a38df1-4acb-4e03-aa78-d95dcbc91eb9/scratchpad")
    items = load(f"{SP}/popQA.tsv")
    picked = sample(items, per_tercile=80)
    kb, held = split_by_entity(picked)
    print(f"loaded {len(items)} triples; sampled {len(picked)} "
          f"(terciles balanced); KB={len(kb)} heldout={len(held)}")
    from collections import Counter
    print("tercile spread:", Counter(i.tercile for i in picked))
    print("relations:", Counter(i.prop for i in picked).most_common(6))
    print("example KB:", kb[0].question, "->", kb[0].answers[:3])
    print("passage:", render_passage(kb[0]))
