"""
relate.py — synonyms, homonyms, senses all live in PROGRAMS in the lattice.

A word is not a single point of meaning. The swarm teaches its relations:
  * SYNONYM : two word-programs that share meaning      -> a relation program
  * SENSE   : one of a word's distinct meanings          -> a sense program
  * HOMONYM : a word with more than one sense            -> emerges from senses

Each is a program node (substrate = the word-programs it relates), so the
lexical web — synonymy, homonymy, and onward (idiom, innuendo) — is all just
more programs in the same lattice. Built by real agents, nothing pre-given.

Run:  python relate.py /tmp/relations.json
"""

from __future__ import annotations
import json
import sys
from store import Store, Node, node_id


def main():
    pos = json.load(open("data/taught_pos_full.json"))
    rel = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/relations.json"))
    store = Store(":memory:")

    cats = set(pos.values()) | {"noun", "verb", "adj"}
    store.put_many([Node("prime", "category", c, agent="seed") for c in cats])
    words = set(rel) | {s for e in rel.values() for s in e.get("synonyms", [])}
    store.put_many([Node("program", "word", w, type=pos.get(w, "noun"), agent="a:teacher",
                        substrate=[node_id("prime", "category",
                                           pos.get(w) if pos.get(w) in cats else "noun")])
                    for w in words])

    syn, senses = [], []
    for w, e in rel.items():
        for s in e.get("synonyms", []):
            syn.append(Node("program", "synonym", f"{w}~{s}", type="relation",
                           substrate=[node_id("program", "word", w),
                                      node_id("program", "word", s)],
                           agent="a:rel", pathways={"a": w, "b": s}))
        for i, sense in enumerate(e.get("senses", [])):
            senses.append(Node("program", "sense", f"{w}#{i}", type="sense",
                              substrate=[node_id("program", "word", w)], agent="a:rel",
                              pathways={"gloss": sense.get("gloss"),
                                        "pos": sense.get("pos")}))
    store.put_many(syn + senses)

    homonyms = {w: e for w, e in rel.items() if len(e.get("senses", [])) > 1}
    print("=" * 66)
    print("LEXICAL RELATIONS — synonyms / homonyms / senses, all as programs")
    print("=" * 66)
    print(f"  words related   : {len(rel)}")
    print(f"  synonym programs: {len(syn)}")
    print(f"  sense programs  : {len(senses)}")
    print(f"  homonyms (>1 sense): {len(homonyms)}")

    print(f"\n  example HOMONYM as programs:")
    for w, e in list(homonyms.items())[:3]:
        print(f"     '{w}':")
        for s in e["senses"]:
            print(f"        - ({s.get('pos')}) {s.get('gloss')}")

    print(f"\n  example SYNONYM programs:")
    for n in syn[:6]:
        print(f"     {n.pathways['a']} ~ {n.pathways['b']}")

    print(f"\n  total lattice: {store.count()} nodes. The lexical web is programs,")
    print("  taught by the swarm — synonymy, homonymy, and onward to innuendo.")


if __name__ == "__main__":
    main()
