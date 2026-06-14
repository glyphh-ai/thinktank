"""
reader.py — one program that handles ALL rules, because rules are DATA.

The thesis, made runnable:
    primes + lattice + (agents at scale)  ->  the complete deterministic system

The reader hardcodes NO grammar. It loads every production from the lattice and
parses with a generic Earley engine (which handles ANY context-free rule, incl.
left-recursive ones like coordination). So:

    every single rule can live in the lattice, and this one program handles them
    all. A swarm adds a rule -> it is a new row -> the reader parses more, with
    zero code change.

We prove it: parse sentences with the seeded grammar, then ADD one rule to the
store at runtime and watch the SAME reader parse a sentence it just couldn't.

Run:  python reader.py
"""

from __future__ import annotations
import json
from store import Store, Node


# ===========================================================================
# Seed a grammar + lexicon INTO THE LATTICE (rules as data, not code).
# ===========================================================================
def seed_grammar(store: Store):
    cats = ["det", "adj", "noun", "pron", "verb", "aux", "prep", "conj"]
    store.put_many([Node("prime", "category", c, agent="a:cat") for c in cats])

    # productions: lhs -> rhs[]  (stored in pathways; value is a unique name)
    rules = [
        ("S",  ["NP", "VP"]),
        ("NP", ["det", "noun"]),
        ("NP", ["det", "adj", "noun"]),
        ("NP", ["pron"]),
        ("NP", ["noun"]),
        ("VP", ["verb"]),
        ("VP", ["verb", "NP"]),
        ("VP", ["aux", "verb"]),
        ("PP", ["prep", "NP"]),
    ]
    store.put_many([_rule_node(lhs, rhs) for lhs, rhs in rules])

    lex = {"the": "det", "a": "det", "big": "adj", "cat": "noun", "dog": "noun",
           "mat": "noun", "she": "pron", "he": "pron", "saw": "verb",
           "sat": "verb", "sleeps": "verb", "sleep": "verb", "on": "prep",
           "and": "conj"}
    store.put_many([Node("program", "word", w, type=pos, agent="a:lex",
                         substrate=[_cat_id(pos)]) for w, pos in lex.items()])


def _cat_id(c):
    from store import node_id
    return node_id("prime", "category", c)


def _rule_node(lhs, rhs) -> Node:
    name = f"{lhs}->{' '.join(rhs)}"
    return Node("program", "rule", name, type="production", agent="a:gram",
                substrate=[_cat_id(s) for s in rhs if s.islower()] or [_cat_id("noun")],
                pathways={"lhs": lhs, "rhs": rhs})


# ===========================================================================
# The reader: loads ALL rules from the lattice, parses with generic Earley.
# ===========================================================================
class Reader:
    def __init__(self, store: Store):
        self.store = store

    def grammar(self) -> dict[str, list[list[str]]]:
        """Every production currently in the lattice. (Re-read each parse, so
        new swarm-written rules take effect immediately.)"""
        g: dict[str, list[list[str]]] = {}
        for (pathways,) in self.store.db.execute(
                "SELECT pathways FROM nodes WHERE kind='rule'"):
            p = json.loads(pathways)
            g.setdefault(p["lhs"], []).append(p["rhs"])
        return g

    def lexicon(self) -> dict[str, str]:
        return {v: t for v, t in self.store.db.execute(
            "SELECT value, type FROM nodes WHERE kind='word'")}

    def parse(self, sentence: str):
        g, lex = self.grammar(), self.lexicon()
        words = sentence.split()
        tags = [lex.get(w, "open") for w in words]
        tree = earley(g, tags, words)
        return tree


# ---- a compact, fully generic Earley parser (handles ANY CFG) --------------
def earley(grammar, tags, words):
    n = len(tags)
    nonterminals = set(grammar)

    # chart[i] = list of states (lhs, rhs(tuple), dot, origin)
    chart = [[] for _ in range(n + 1)]

    def add(i, state):
        if state not in chart[i]:
            chart[i].append(state)

    for rhs in grammar.get("S", []):
        add(0, ("S", tuple(rhs), 0, 0))

    for i in range(n + 1):
        k = 0
        while k < len(chart[i]):
            lhs, rhs, dot, origin = chart[i][k]
            k += 1
            if dot < len(rhs):
                nxt = rhs[dot]
                if nxt in nonterminals:                       # PREDICT
                    for r in grammar[nxt]:
                        add(i, (nxt, tuple(r), 0, i))
                elif i < n and tags[i] == nxt:                # SCAN
                    add(i + 1, (lhs, rhs, dot + 1, origin))
            else:                                             # COMPLETE
                for st in chart[origin]:
                    slhs, srhs, sdot, sorig = st
                    if sdot < len(srhs) and srhs[sdot] == lhs:
                        add(i, (slhs, srhs, sdot + 1, sorig))

    if not any(s == ("S", s[1], len(s[1]), 0) for s in chart[n] if s[0] == "S"):
        return None

    # completed states ending at j: for tree reconstruction
    completed = {}
    for j in range(n + 1):
        for lhs, rhs, dot, origin in chart[j]:
            if dot == len(rhs):
                completed.setdefault((lhs, origin), []).append((rhs, j))

    def build(sym, i, j):
        if sym not in nonterminals:                           # terminal (POS)
            return (words[i],) if (j == i + 1 and tags[i] == sym) else None
        for rhs, end in completed.get((sym, i), []):
            if end != j:
                continue
            kids = _segment(rhs, i, j, build)
            if kids is not None:
                return (sym, *kids)
        return None

    return build("S", 0, n)


def _segment(rhs, i, j, build):
    if not rhs:
        return [] if i == j else None
    first, rest = rhs[0], rhs[1:]
    for k in range(i, j + 1):
        left = build(first, i, k)
        if left is None:
            continue
        right = _segment(rest, k, j, build)
        if right is not None:
            return [left] + right
    return None


def render(tree) -> str:
    if isinstance(tree, str):
        return tree
    if len(tree) == 1 or (len(tree) == 2 and isinstance(tree[1], str)
                          and tree[0] not in ("S", "NP", "VP", "PP")):
        return tree[0]
    head, kids = tree[0], tree[1:]
    inner = " ".join(render(k) if not isinstance(k, str) else k for k in kids)
    return f"[{head} {inner}]"


# ===========================================================================
# DEMO
# ===========================================================================
def main():
    store = Store(":memory:")
    seed_grammar(store)
    reader = Reader(store)

    print("=" * 66)
    print("ONE PROGRAM, ALL RULES — grammar read from the lattice")
    print("=" * 66)
    g = reader.grammar()
    print(f"  rules in lattice: {sum(len(v) for v in g.values())}"
          f"  across {len(g)} categories\n")
    for s in ["the big cat saw a dog", "she sleeps", "the cat and the dog sleep"]:
        t = reader.parse(s)
        print(f"  {s}")
        print(f"    -> {render(t) if t else 'NO PARSE (no rule covers it yet)'}")

    print("\n" + "-" * 66)
    print("  SWARM ADDS ONE RULE -> same program parses more, zero code change")
    print("  adding (left-recursive!):  NP -> NP conj NP")
    store.put_many([_rule_node("NP", ["NP", "conj", "NP"])])

    s = "the cat and the dog sleep"
    t = reader.parse(s)
    print(f"\n  {s}")
    print(f"    -> {render(t) if t else 'NO PARSE'}")
    print("\n  => every rule can live in the lattice; this one generic reader")
    print("     handles them all (Earley parses any CFG, even left recursion).")
    print("     The swarm writes rules as data; reading is deterministic.")


if __name__ == "__main__":
    main()
