"""
comprehend.py — the comprehension SYSTEM: the inverse of the reader.

The reader EXPANDS productions to turn form into structure (top-down).
Comprehension INVERTS them to turn structure into meaning (bottom-up): for
every syntactic production the swarm built, there is a paired SEMANTIC RULE --
its inverse -- and one generic interpreter composes meaning over the parse.
(This is compositional semantics: the meaning of the whole is a function of the
meanings of the parts and the rule that joined them.)

It is a system of its own:
  * NEW primes    : sem-primes (SEE, SLEEP, BIG, KIND...) and roles (AGENT...)
  * SHARED primes : the words, categories, and PRODUCTIONS already in the lattice
  * programs      : lexical meanings (word -> sem-primes) and one semantic rule
                    paired to each production (the inverse)
  * system        : 'comprehension', whose substrate is those programs

And -- the point you made -- the swarm builds this the same way it built the
grammar: it just builds the INVERSE of each rule. Same pipeline, new tower.

Run:  python comprehend.py
"""

from __future__ import annotations
import json
from store import Store, Node, node_id
import reader as R


# ===========================================================================
# Seed the comprehension tower INTO THE LATTICE (new + shared primes).
# ===========================================================================
SEM_PRIMES = ["SEE", "SLEEP", "SIT", "BIG", "KIND", "LIVE", "SMALL", "MOVE",
              "ANIMATE", "MANY"]
ROLES = ["agent", "patient"]

# lexical meanings: word -> a meaning object (swarm-built data).
LEXSEM = {
    "cat":  {"t": "concept", "label": "cat", "sem": ["KIND", "LIVE", "SMALL", "MOVE"]},
    "dog":  {"t": "concept", "label": "dog", "sem": ["KIND", "LIVE", "MOVE"]},
    "mat":  {"t": "concept", "label": "mat", "sem": ["KIND"]},
    "she":  {"t": "concept", "label": "she", "sem": ["ANIMATE"]},
    "he":   {"t": "concept", "label": "he", "sem": ["ANIMATE"]},
    "big":  {"t": "mod", "sem": ["BIG"]},
    "saw":  {"t": "verb", "name": "SEE", "roles": ["agent", "patient"]},
    "sat":  {"t": "verb", "name": "SIT", "roles": ["agent"]},
    "sleep":  {"t": "verb", "name": "SLEEP", "roles": ["agent"]},
    "sleeps": {"t": "verb", "name": "SLEEP", "roles": ["agent"]},
}

# semantic rules: one per production = the INVERSE. Keyed by (lhs, rhs).
SEMRULES = {
    ("S",  ("NP", "VP")):          {"op": "saturate", "pred": 1, "arg": 0, "role": "agent"},
    ("VP", ("verb", "NP")):        {"op": "predicate", "head": 0, "args": [[1, "patient"]]},
    ("VP", ("verb",)):             {"op": "predicate", "head": 0, "args": []},
    ("VP", ("aux", "verb")):       {"op": "predicate", "head": 1, "args": []},
    ("NP", ("det", "noun")):       {"op": "refer", "head": 1, "det": 0},
    ("NP", ("det", "adj", "noun")):{"op": "refer", "head": 2, "mods": [1], "det": 0},
    ("NP", ("noun",)):             {"op": "refer", "head": 0},
    ("NP", ("pron",)):             {"op": "refer", "head": 0},
    ("NP", ("NP", "conj", "NP")):  {"op": "join", "parts": [0, 2]},
}


def seed_comprehension(store: Store):
    # new primes: sem-primes + role primes
    store.put_many([Node("prime", "sem-prime", p, agent="a:sem") for p in SEM_PRIMES]
                   + [Node("prime", "role", r, agent="a:sem") for r in ROLES])
    # programs: lexical meanings, substrate = the sem-primes they use (shared)
    lex_nodes = []
    for w, m in LEXSEM.items():
        sub = [node_id("prime", "sem-prime", s) for s in m.get("sem", [])] or \
              [node_id("prime", "sem-prime", "KIND")]
        lex_nodes.append(Node("program", "lex-sem", w, type="meaning",
                              substrate=sub, agent="a:sem",
                              pathways={"meaning": m}))
    store.put_many(lex_nodes)
    # programs: one semantic rule per production -- substrate includes the
    # GRAMMAR PRODUCTION it inverts (the pairing lives in the lattice).
    sem_nodes = []
    for (lhs, rhs), spec in SEMRULES.items():
        prod_name = f"{lhs}->{' '.join(rhs)}"
        prod_id = node_id("program", "rule", prod_name)
        roles = [node_id("prime", "role", r) for r in ROLES if r in json.dumps(spec)]
        sem_nodes.append(Node("program", "semrule", prod_name, type="inverse",
                              substrate=[prod_id] + roles, agent="a:sem",
                              pathways={"lhs": lhs, "rhs": list(rhs), "spec": spec}))
    store.put_many(sem_nodes)
    # the system
    store.put_many([Node("system", "comprehension", "core", type="meaning",
                         substrate=[n.id for n in lex_nodes + sem_nodes],
                         agent="a:sem")])


# ===========================================================================
# The comprehender: reads sem-rules from the lattice, composes meaning.
# ===========================================================================
class Comprehender:
    def __init__(self, store: Store):
        self.store = store
        self.lex = {v: t for v, t in store.db.execute(
            "SELECT value, type FROM nodes WHERE kind='word'")}
        self.lexsem = {v: json.loads(p)["meaning"] for v, p in store.db.execute(
            "SELECT value, pathways FROM nodes WHERE kind='lex-sem'")}
        self.semrules = {}
        for v, p in store.db.execute(
                "SELECT value, pathways FROM nodes WHERE kind='semrule'"):
            d = json.loads(p)
            self.semrules[(d["lhs"], tuple(d["rhs"]))] = d["spec"]
        self.NT = {lhs for (lhs, _rhs) in self.semrules}

    def meaning_of(self, word: str) -> dict:
        return self.lexsem.get(word, {"t": "fn", "label": word})

    def _sym(self, node) -> str:
        return node[0] if node[0] in self.NT else self.lex.get(node[0], "open")

    def comprehend(self, node):
        if node[0] not in self.NT:                       # terminal: a word
            return self.meaning_of(node[0])
        lhs, kids = node[0], list(node[1:])
        rhs = tuple(self._sym(k) for k in kids)
        spec = self.semrules[(lhs, rhs)]
        cms = [self.comprehend(k) for k in kids]
        return apply_spec(spec, cms, kids)


def apply_spec(spec, cms, kids):
    op = spec["op"]
    if op == "refer":
        head = dict(cms[spec["head"]])
        head["sem"] = list(head.get("sem", []))
        for mi in spec.get("mods", []):
            head["sem"] = cms[mi].get("sem", []) + head["sem"]
        if "det" in spec:
            head["det"] = kids[spec["det"]][0]
        return head
    if op == "predicate":
        v = cms[spec["head"]]
        pred = {"t": "pred", "name": v["name"], "roles": {}}
        for ci, role in spec.get("args", []):
            pred["roles"][role] = cms[ci]
        return pred
    if op == "saturate":
        pred = dict(cms[spec["pred"]])
        pred["roles"] = dict(pred["roles"])
        pred["roles"][spec["role"]] = cms[spec["arg"]]
        return pred
    if op == "join":
        return {"t": "set", "items": [cms[i] for i in spec["parts"]]}
    return {"t": "?", "op": op}


ROLE_ORDER = ["agent", "patient", "theme"]


def render(m) -> str:
    t = m.get("t")
    if t == "pred":
        rs = sorted(m["roles"], key=lambda r: ROLE_ORDER.index(r)
                    if r in ROLE_ORDER else 9)
        return f"{m['name']}(" + ", ".join(f"{r}={render(m['roles'][r])}"
                                           for r in rs) + ")"
    if t == "concept":
        sem = ",".join(m.get("sem", []))
        return f"{m['label']}[{sem}]" if sem else m["label"]
    if t == "set":
        return "{" + ", ".join(render(i) for i in m["items"]) + "}"
    if t == "mod":
        return ",".join(m.get("sem", []))
    return m.get("label", "?")


# ===========================================================================
# DEMO
# ===========================================================================
def main():
    store = Store(":memory:")
    R.seed_grammar(store)
    store.put_many([R._rule_node("NP", ["NP", "conj", "NP"])])   # coordination
    seed_comprehension(store)

    reader = R.Reader(store)
    comp = Comprehender(store)

    print("=" * 68)
    print("COMPREHENSION SYSTEM — the inverse of the reader, over the lattice")
    print("=" * 68)
    print("  reader  EXPANDS productions: form -> structure")
    print("  comprehend INVERTS them:     structure -> meaning")
    print(f"  paired semantic rules in lattice: {len(comp.semrules)}"
          f"  (one inverse per production)\n")

    for s in ["the big cat saw a dog", "she sleeps",
              "the cat and the dog sleep"]:
        tree = reader.parse(s)
        meaning = comp.comprehend(tree)
        print(f"  {s}")
        print(f"    structure: {R.render(tree)}")
        print(f"    meaning  : {render(meaning)}\n")

    print("-" * 68)
    print("  INVERSE PAIRING lives in the lattice: each semrule's substrate")
    print("  includes the grammar production it inverts. e.g. VP -> verb NP:")
    prod = node_id("program", "rule", "VP->verb NP")
    sem = node_id("program", "semrule", "VP->verb NP")
    print(f"     production {prod[:14]}..  <->  semrule {sem[:14]}..")
    print("\n  => the swarm built the grammar; the swarm builds its inverse the")
    print("     same way. New tower, same pipeline: primes -> programs -> system.")


if __name__ == "__main__":
    main()
