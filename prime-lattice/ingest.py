"""
ingest.py — load committed knowledge into the persistent lattice (idempotent).

Accumulates into one durable lattice.db: the lexicon (taught POS), the verb
frames, and the Wikipedia knowledge graph. Run it again and it's a no-op
(content-ID UPSERT). Ingest new data and it grows. Nothing is thrown away.

Run:  python ingest.py
"""

from __future__ import annotations
import json
from lattice import open_lattice, stats
from store import Node, node_id
import wiki_brain

CLASSES = ["ANIMATE", "PHYSICAL", "PLACE", "ABSTRACT", "COMMUNICATION", "BODY", "TIME"]


def ingest(store):
    # --- lexicon: taught POS for the whole corpus vocabulary ---
    pos = json.load(open("data/taught_pos_full.json"))
    store.put_many([Node("prime", "category", c, agent="seed") for c in set(pos.values())])
    store.put_many([Node("program", "word", w, type=p, agent="ingest",
                        substrate=[node_id("prime", "category", p)])
                    for w, p in pos.items()])

    # --- frames: verb selectional restrictions ---
    store.put_many([Node("prime", "class", c, agent="seed") for c in CLASSES])
    frames = json.load(open("data/meaning_compounding.json"))["frames1"]
    store.put_many([Node("program", "frame", v, type="predicate", agent="ingest",
                        substrate=[node_id("prime", "class", sig[0]) for sig in sigs
                                   if sig[0] in CLASSES] or [node_id("prime", "category", "verb")],
                        pathways={"sigs": sigs})
                    for v, sigs in frames.items() if v != "don"])

    # --- knowledge graph: Wikipedia facts ---
    wiki_brain.build(store, json.load(open("data/wiki_facts.json")))


def main():
    store = open_lattice()
    before = stats(store)
    ingest(store)
    after = stats(store)
    print("=" * 60)
    print("INGEST -> persistent lattice (lattice.db)")
    print("=" * 60)
    print(f"  nodes before: {before['total']:>6}")
    print(f"  nodes after : {after['total']:>6}   (+{after['total'] - before['total']})")
    print(f"  by kind: {after['by_kind']}")
    print("\n  re-run = no-op (idempotent). ingest new data = it grows.")


if __name__ == "__main__":
    main()
