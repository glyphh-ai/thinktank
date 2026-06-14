"""
ingest.py — load committed knowledge into the persistent lattice (idempotent).

Accumulates into one durable lattice.db:
  * lexicon     — taught POS for the whole vocabulary
  * generation  — 867 verb frames, 954 noun classes, 321 adjectives,
                  productions, realization rules (the GENERATION SYSTEM)
  * knowledge   — the Wikipedia fact graph

Re-run = no-op (content-ID UPSERT). Ingest new data = it grows. Nothing thrown
away. The generation system lives here as programs; gensystem.py just reads it.

Run:  python ingest.py
"""

from __future__ import annotations
import json
from lattice import open_lattice, stats
from store import Node, node_id
import wiki_brain

CLASSES = ["ANIMATE", "PHYSICAL", "PLACE", "ABSTRACT", "COMMUNICATION", "BODY", "TIME", "EVENT"]
CATS = ["det", "noun", "verb", "adj", "prep"]
PRODS = [("S", ["NP", "VP"]), ("NP", ["det", "noun"]), ("NP", ["det", "adj", "noun"]),
         ("NP", ["noun"]), ("VP", ["verb"]), ("VP", ["verb", "NP"]),
         ("VP", ["verb", "PP"]), ("PP", ["prep", "NP"])]
REAL = [("proper", "{w}"), ("PLACE", "to the {w}"), ("default", "the {w}")]


def ingest_lexicon(store):
    pos = json.load(open("data/taught_pos_full.json"))
    store.put_many([Node("prime", "category", c, agent="seed")
                    for c in set(pos.values()) | set(CATS)])
    store.put_many([Node("program", "word", w, type=p, agent="ingest",
                        substrate=[node_id("prime", "category", p)])
                    for w, p in pos.items()])


def ingest_generation(store):
    store.put_many([Node("prime", "class", c, agent="seed") for c in CLASSES]
                   + [Node("prime", "category", c, agent="seed") for c in CATS])

    ncls = json.load(open("data/gen_nounclass.json"))
    store.put_many([Node("program", "nclass", w, type="typing", agent="gen",
                        substrate=[node_id("prime", "class", c)], pathways={"class": c})
                    for w, c in ncls.items() if c in CLASSES])

    adj = json.load(open("data/gen_adjclass.json"))
    store.put_many([Node("program", "adjmod", w, type="modifier", agent="gen",
                        substrate=[node_id("prime", "category", "adj")],
                        pathways={"modifies": [c for c in cs if c in CLASSES]})
                    for w, cs in adj.items() if isinstance(cs, list)])

    frames = json.load(open("data/gen_frames.json"))
    store.put_many([Node("program", "frame", v, type="predicate", agent="gen",
                        substrate=[node_id("prime", "class", s) for s in fr.get("subj", [])
                                   if s in CLASSES] or [node_id("prime", "category", "verb")],
                        pathways={"subj": fr.get("subj", []), "obj": fr.get("obj", []),
                                  "past": fr.get("past", v)})
                    for v, fr in frames.items() if isinstance(fr, dict)])

    store.put_many([Node("program", "production", f"{l}->{' '.join(r)}", type="rule",
                        agent="gen", substrate=[node_id("prime", "category", s)
                                                for s in r if s.islower()]
                                  or [node_id("prime", "category", "noun")],
                        pathways={"lhs": l, "rhs": r}) for l, r in PRODS])
    store.put_many([Node("program", "realize", t, type="surface", agent="gen",
                        substrate=[node_id("prime", "category", "det")],
                        pathways={"template": tmpl}) for t, tmpl in REAL])

    ids = [r[0] for r in store.db.execute(
        "SELECT id FROM nodes WHERE kind IN ('frame','nclass','adjmod','production','realize')")]
    store.put_many([Node("system", "generation", "core", type="generator",
                        substrate=ids, agent="gen")])


def ingest_relations(store):
    rels = json.load(open("data/gen_relations.json"))
    store.put_many([Node("prime", "sem-prime", "REL", agent="seed")])
    syn, hyp = [], []
    for noun, d in rels.items():
        if not isinstance(d, dict):
            continue
        for a in d.get("syn", []):
            syn.append(Node("program", "synonym", f"{a}=>{noun}", type="relation",
                           agent="rel", substrate=[node_id("prime", "sem-prime", "REL")],
                           pathways={"alias": a, "canonical": noun}))
        if d.get("hyper"):
            hyp.append(Node("program", "hypernym", noun, type="relation",
                           agent="rel", substrate=[node_id("prime", "sem-prime", "REL")],
                           pathways={"hyper": d["hyper"]}))
    store.put_many(syn + hyp)


def ingest_inference(store):
    store.put_many([Node("prime", "sem-prime", "RULE", agent="seed")])
    rules = json.load(open("data/inference_rules.json"))
    store.put_many([Node("program", "inference", r["name"], type="rule", agent="inf",
                        substrate=[node_id("prime", "sem-prime", "RULE")],
                        pathways={"rule": r}) for r in rules])


def main():
    store = open_lattice()
    before = stats(store)
    ingest_lexicon(store)
    ingest_generation(store)
    ingest_relations(store)
    ingest_inference(store)
    wiki_brain.build(store, json.load(open("data/wiki_facts.json")))
    wiki_brain.build(store, json.load(open("data/wiki_astro_facts.json")))
    after = stats(store)
    print("=" * 64)
    print("INGEST -> persistent lattice (lattice.db)")
    print("=" * 64)
    print(f"  nodes: {before['total']} -> {after['total']}  (+{after['total']-before['total']})")
    print(f"  generation system: "
          f"{after['by_kind'].get('frame',0)} frames, "
          f"{after['by_kind'].get('nclass',0)} noun-classes, "
          f"{after['by_kind'].get('adjmod',0)} adjectives, "
          f"{after['by_kind'].get('production',0)} productions")
    print(f"  by kind: {after['by_kind']}")


if __name__ == "__main__":
    main()
