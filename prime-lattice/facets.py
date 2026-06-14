"""
facets.py — all of language falls into ONE lattice.

Sounds->words was a single facet of a single tower. A word is not one node; it
is the JOIN of many towers, each a different pathway over shared primes:

    orthography : letters        c a t
    phonology   : sounds         /k/ /æ/ /t/
    morphology  : morphemes      cat(root)  -s(plural)
    syntax      : categories     noun
    semantics   : sem-primes     KIND LIVE SMALL MOVE      (NSM-style)

The same word participates in all of them at once -> the same atom under many
parents -> a lattice, not a tree. "All language falls inline" means every one
of these dimensions is just more nodes + pathways in the same graph, validated
by the same schema-on-write store. Semantics is the foundational tower: it is
what makes comprehension (not just spelling) possible.

Run:  python facets.py
"""

from __future__ import annotations
import collections
from store import Store, Node, node_id


# ---------------------------------------------------------------------------
# Seed the prime atoms for every tower (each tower = a kind of prime).
# ---------------------------------------------------------------------------
def seed(store: Store):
    nodes = []
    nodes += [Node("prime", "letter", c, agent="a:ortho") for c in "cats"]
    nodes += [Node("prime", "sound", s, agent="a:phon")
              for s in ["/k/", "/æ/", "/t/", "/s/"]]
    nodes += [Node("prime", "morpheme", m, agent="a:morph",
                   pathways={"gloss": g})
              for m, g in [("cat", "root:feline"), ("-s", "suffix:plural")]]
    nodes += [Node("prime", "category", c, agent="a:syntax") for c in ["noun"]]
    nodes += [Node("prime", "sem-prime", p, agent="a:sem")
              for p in ["KIND", "LIVE", "SMALL", "MOVE", "MANY"]]
    store.put_many(nodes)


def P(kind, value):                       # id of a prime atom
    return node_id("prime", kind, value)


# ---------------------------------------------------------------------------
# A word program whose substrate spans ALL towers at once.
# ---------------------------------------------------------------------------
def word_cat() -> Node:
    return Node("program", "word", "cat", type="noun", agent="a:lex",
                substrate=[
                    P("letter", "c"), P("letter", "a"), P("letter", "t"),
                    P("sound", "/k/"), P("sound", "/æ/"), P("sound", "/t/"),
                    P("morpheme", "cat"),
                    P("category", "noun"),
                    P("sem-prime", "KIND"), P("sem-prime", "LIVE"),
                    P("sem-prime", "SMALL"), P("sem-prime", "MOVE"),
                ])


def word_cats() -> Node:
    return Node("program", "word", "cats", type="noun", agent="a:lex",
                substrate=[
                    P("letter", "c"), P("letter", "a"), P("letter", "t"), P("letter", "s"),
                    P("sound", "/k/"), P("sound", "/æ/"), P("sound", "/t/"), P("sound", "/s/"),
                    P("morpheme", "cat"), P("morpheme", "-s"),         # shares the root
                    P("category", "noun"),
                    P("sem-prime", "KIND"), P("sem-prime", "LIVE"),
                    P("sem-prime", "SMALL"), P("sem-prime", "MOVE"),
                    P("sem-prime", "MANY"),                            # plural adds MANY
                ])


TOWER = {"letter": "orthography", "sound": "phonology", "morpheme": "morphology",
         "category": "syntax", "sem-prime": "semantics"}


def facets_of(store: Store, word_id: str) -> dict:
    """Reconstruct a word's towers = its substrate grouped by child kind."""
    by_tower = collections.defaultdict(list)
    for cid in store.substrate_of(word_id):
        kind, value = store.db.execute(
            "SELECT kind, value FROM nodes WHERE id=?", (cid,)).fetchone()
        by_tower[TOWER[kind]].append(value)
    return by_tower


def main():
    store = Store(":memory:")
    seed(store)
    cat, cats = word_cat(), word_cats()
    store.put_many([cat, cats])

    print("=" * 66)
    print("ALL LANGUAGE FALLS INLINE — one word, every tower, one lattice")
    print("=" * 66)
    for w in (cat, cats):
        print(f"\n  word '{w.value}'  (typed: {w.type})")
        for tower in ("orthography", "phonology", "morphology", "syntax", "semantics"):
            vals = facets_of(store, w.id).get(tower, [])
            print(f"     {tower:11}: {' '.join(vals)}")

    print("\n" + "-" * 66)
    print("  MULTI-INHERITANCE — shared primes connect words ACROSS towers:")
    root = P("morpheme", "cat")
    kind = P("sem-prime", "KIND")
    print(f"     morpheme 'cat'  used by: {[_v(store,i) for i in store.users_of(root)]}")
    print(f"     sem-prime KIND  used by: {[_v(store,i) for i in store.users_of(kind)]}")
    print("\n  => 'cat' and 'cats' join at the root morpheme AND the meaning,")
    print("     not just at letters. Every linguistic dimension is a pathway in")
    print("     the same graph. Phonics was one tower; meaning is the deepest.")


def _v(store, node_id):
    return store.db.execute("SELECT value FROM nodes WHERE id=?", (node_id,)).fetchone()[0]


if __name__ == "__main__":
    main()
