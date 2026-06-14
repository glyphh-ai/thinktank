"""
learn.py — there is no lexicon. We TEACH every word as we meet it.

The thesis: nothing is pre-supplied. The lattice starts with grammar rules and
a handful of function words. When a sentence contains words it has never seen,
the swarm TEACHES them — new primes, new word-programs, new meanings — straight
into the lattice at build time. Then the fixed deterministic towers understand
the sentence. No external lexicon. No model at runtime.

Loop:
  sentence -> which words are UNKNOWN?  -> swarm teaches them into the lattice
           -> deterministic parse + comprehend now succeeds

Run:  python learn.py                      # show the gap (untaught words)
      python learn.py --taught taught.json # teach them, then understand
"""

from __future__ import annotations
import json
import sys
import reader as R
import comprehend as C
import emotion as E
from store import Store, Node, node_id


def seed_base() -> Store:
    """A bare mind: grammar rules + a few function words. No content words."""
    store = Store(":memory:")
    cats = ["det", "adj", "noun", "pron", "verb", "aux", "prep", "conj"]
    store.put_many([Node("prime", "category", c, agent="seed") for c in cats])
    rules = [("S", ["NP", "VP"]), ("NP", ["det", "noun"]),
             ("NP", ["det", "adj", "noun"]), ("NP", ["pron"]), ("NP", ["noun"]),
             ("VP", ["verb"]), ("VP", ["verb", "NP"]), ("VP", ["aux", "verb"]),
             ("PP", ["prep", "NP"])]
    store.put_many([R._rule_node(l, r) for l, r in rules])
    fw = {"the": "det", "a": "det", "to": "prep", "and": "conj"}
    store.put_many([Node("program", "word", w, type=p, agent="seed",
                        substrate=[R._cat_id(p)]) for w, p in fw.items()])
    # comprehension: fixed sem-rules (the frozen interpreter core)
    store.put_many([Node("prime", "role", r, agent="seed") for r in C.ROLES])
    for (lhs, rhs), spec in C.SEMRULES.items():
        prod = node_id("program", "rule", f"{lhs}->{' '.join(rhs)}")
        if store.db.execute("SELECT 1 FROM nodes WHERE id=?", (prod,)).fetchone():
            store.put_many([Node("program", "semrule", f"{lhs}->{' '.join(rhs)}",
                                type="inverse",
                                substrate=[prod] + [node_id("prime", "role", r)
                                                    for r in C.ROLES if r in json.dumps(spec)],
                                agent="seed",
                                pathways={"lhs": lhs, "rhs": list(rhs), "spec": spec})])
    E.seed_emotion(store)
    return store


def known_words(store):
    return {v for (v,) in store.db.execute("SELECT value FROM nodes WHERE kind='word'")}


def unknown_in(store, sentence):
    have = known_words(store)
    return [w for w in sentence.rstrip(".").lower().split() if w not in have]


def teach(store, taught: dict):
    """Build new PRIMES, WORD-programs and MEANINGS into the lattice. This is
    the whole thesis: hardcode/teach exactly what's needed, as we go."""
    sems = set()
    for e in taught.values():
        sems |= set(e["meaning"].get("sem", []))
    store.put_many([Node("prime", "sem-prime", s, agent="a:teacher") for s in sems])  # new primes
    nodes = []
    for w, e in taught.items():
        nodes.append(Node("program", "word", w, type=e["pos"], agent="a:teacher",
                         substrate=[R._cat_id(e["pos"])]))
        m = e["meaning"]
        sub = [node_id("prime", "sem-prime", s) for s in m.get("sem", [])] or \
              [R._cat_id(e["pos"])]
        nodes.append(Node("program", "lex-sem", w, type="meaning", substrate=sub,
                         agent="a:teacher", pathways={"meaning": m}))
    store.put_many(nodes)


def understand(store, sentence):
    tree = R.Reader(store).parse(sentence.rstrip("."))
    if not tree:
        return None, None
    return tree, C.Comprehender(store).comprehend(tree)


def main():
    sentence = "the clever fox chased the rabbit"
    store = seed_base()
    n0 = store.count()

    print("=" * 64)
    print("NO LEXICON — we teach the words as we meet them")
    print("=" * 64)
    print(f"  bare mind: {n0} nodes (grammar rules + function words only)")
    print(f"  sentence : \"{sentence}\"")

    unknown = unknown_in(store, sentence)
    tree, _ = understand(store, sentence)
    print(f"\n  BEFORE teaching:")
    print(f"     unknown words : {unknown}")
    print(f"     can understand: {'yes' if tree else 'NO — words not taught yet'}")
    json.dump(unknown, open("/tmp/unknown.json", "w"))

    if len(sys.argv) > 2 and sys.argv[1] == "--taught":
        taught = json.load(open(sys.argv[2]))
        teach(store, taught)                          # the swarm teaches them in
        tree, meaning = understand(store, sentence)
        print(f"\n  the swarm taught {len(taught)} words "
              f"(+{store.count() - n0} new nodes: primes, words, meanings)")
        print(f"  AFTER teaching:")
        print(f"     structure : {R.render(tree) if tree else 'NO'}")
        print(f"     meaning   : {C.render(meaning) if meaning else 'NO'}")
        print(f"\n  => parsed and comprehended DETERMINISTICALLY, no model at")
        print("     runtime. The lattice grew by exactly what was needed.")
    else:
        print(f"\n  -> wrote unknown words to /tmp/unknown.json for the swarm to teach.")


if __name__ == "__main__":
    main()
