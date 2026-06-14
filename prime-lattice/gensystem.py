"""
gensystem.py — GENERATION as a system in the lattice (primes -> programs -> system).

Not a one-off. The generation KNOWLEDGE lives in the lattice as programs; the
generator is a small generic interpreter that reads them and walks a tree --
exactly like the Earley reader reads productions. Add a frame/word/rule to the
lattice and generation changes with zero code change.

PRIMES   : category (det,noun,verb,prep) ; class (ANIMATE,PHYSICAL,PLACE,...) ; role
PROGRAMS : word(+class) ; production(S->NP VP) ; frame(verb->selectional sigs) ;
           inflect(lemma->past) ; realize(class->surface template)
SYSTEM   : generation -> substrate = all of the above
TREE     : generate = frame -> proposition -> syntactic tree -> realized surface

Run:  python gensystem.py
"""

from __future__ import annotations
import collections
import json
import random
from store import Store, Node, node_id
from reader import earley

random.seed(5)

# ---------------------------------------------------------------------------
# BUILD THE GENERATION SYSTEM IN THE LATTICE (all knowledge = program nodes)
# ---------------------------------------------------------------------------
def build(store: Store):
    cats = ["det", "noun", "verb", "prep"]
    classes = ["ANIMATE", "PHYSICAL", "PLACE", "ABSTRACT", "COMMUNICATION", "BODY", "TIME"]
    store.put_many([Node("prime", "category", c, agent="g") for c in cats]
                   + [Node("prime", "class", c, agent="g") for c in classes])

    # words: nouns carry their semantic class; function words are categories
    noun_class = json.load(open("/tmp/noun_class.json"))
    wn = []
    for w, c in noun_class.items():
        wn.append(Node("program", "word", w, type="noun", agent="g",
                      substrate=[node_id("prime", "class", c)],
                      pathways={"class": c, "proper": w in ("alice", "dinah")}))
    wn += [Node("program", "word", w, type="det", agent="g",
               substrate=[node_id("prime", "category", "det")]) for w in ("the", "a")]
    store.put_many(wn)

    # productions (structure programs)
    prods = [("S", ["NP", "VP"]), ("NP", ["det", "noun"]), ("NP", ["noun"]),
             ("VP", ["verb", "NP"]), ("VP", ["verb"]), ("VP", ["verb", "PP"]),
             ("PP", ["prep", "NP"])]
    store.put_many([Node("program", "production", f"{l}->{' '.join(r)}", type="rule",
                        agent="g", substrate=[node_id("prime", "category", s)
                                              for s in r if s.islower()] or
                                  [node_id("prime", "category", "noun")],
                        pathways={"lhs": l, "rhs": r}) for l, r in prods])

    # frames (selectional programs) from the measured corpus-1 frames
    frames = json.load(open("data/meaning_compounding.json"))["frames1"]
    store.put_many([Node("program", "frame", v, type="predicate", agent="g",
                        substrate=[node_id("prime", "class", sig[0])
                                   for sig in sigs if sig[0] in
                                   ("ANIMATE","PHYSICAL","PLACE","ABSTRACT","COMMUNICATION","BODY","TIME")]
                                  or [node_id("prime", "category", "verb")],
                        pathways={"sigs": sigs})
                    for v, sigs in frames.items() if v != "don"])

    # inflection programs (lemma -> past surface)
    infl = {"know": "knew", "come": "came", "think": "thought", "see": "saw",
            "get": "got", "go": "went", "tell": "told", "said": "said",
            "won": "won", "went": "went", "got": "got"}
    store.put_many([Node("program", "inflect", v, type="morph", agent="g",
                        substrate=[node_id("prime", "category", "verb")],
                        pathways={"past": p}) for v, p in infl.items()])

    # realization programs (class -> surface template)
    real = [("proper", "{w}"), ("PLACE", "to the {w}"), ("default", "the {w}")]
    store.put_many([Node("program", "realize", trig, type="surface", agent="g",
                        substrate=[node_id("prime", "category", "det")],
                        pathways={"trigger": trig, "template": tmpl}) for trig, tmpl in real])

    # the system: substrate = all the generation programs
    ids = [r[0] for r in store.db.execute(
        "SELECT id FROM nodes WHERE kind IN "
        "('word','production','frame','inflect','realize')")]
    store.put_many([Node("system", "generation", "core", type="generator",
                        substrate=ids, agent="g")])


# ---------------------------------------------------------------------------
# THE GENERIC GENERATOR — reads the system from the lattice, walks a tree
# ---------------------------------------------------------------------------
class Generator:
    def __init__(self, store: Store):
        self.store = store
        self.grammar = collections.defaultdict(list)
        for lhs, pathways in store.db.execute(
                "SELECT value, pathways FROM nodes WHERE kind='production'"):
            p = json.loads(pathways); self.grammar[p["lhs"]].append(tuple(p["rhs"]))
        self.frames = {v: json.loads(p)["sigs"] for v, p in store.db.execute(
            "SELECT value, pathways FROM nodes WHERE kind='frame'")}
        self.inflect = {v: json.loads(p)["past"] for v, p in store.db.execute(
            "SELECT value, pathways FROM nodes WHERE kind='inflect'")}
        self.realize_rules = {json.loads(p)["trigger"]: json.loads(p)["template"]
                              for _, p in store.db.execute(
                              "SELECT value, pathways FROM nodes WHERE kind='realize'")}
        self.by_class = collections.defaultdict(list)
        self.proper = set()
        for w, pathways in store.db.execute(
                "SELECT value, pathways FROM nodes WHERE kind='word' AND type='noun'"):
            p = json.loads(pathways); self.by_class[p["class"]].append(w)
            if p.get("proper"): self.proper.add(w)

    def realize_np(self, word, cls):
        if word in self.proper:
            tmpl = self.realize_rules["proper"]
        elif cls == "PLACE" and "PLACE" in self.realize_rules:
            tmpl = self.realize_rules["PLACE"]
        else:
            tmpl = self.realize_rules["default"]
        return ("NP", *tmpl.format(w=word).split())

    def has(self, lhs, rhs):
        return tuple(rhs) in self.grammar[lhs]

    def generate(self):
        """frame -> proposition -> tree -> surface, all from the lattice."""
        v = random.choice([v for v in self.frames if v in self.inflect])
        subj_cls, obj_cls = random.choice(self.frames[v])
        if subj_cls not in self.by_class:
            return None
        subj = random.choice(self.by_class[subj_cls])
        vb = self.inflect[v]
        np_subj = self.realize_np(subj, subj_cls)
        if obj_cls in self.by_class and self.has("VP", ["verb", "NP"]):
            obj = random.choice(self.by_class[obj_cls])
            vp = ("VP", ("verb", vb), self.realize_np(obj, obj_cls))
        elif self.has("VP", ["verb"]):
            vp = ("VP", ("verb", vb))
        else:
            return None
        return ("S", np_subj, vp)


def leaves(tree):
    out = []
    def w(t):
        if isinstance(t, tuple) and t and isinstance(t[0], str) and t[0].isupper():
            for k in t[1:]: w(k)
        elif isinstance(t, tuple) and len(t) == 2:  # (cat, word) leaf
            out.append(t[1])
        elif isinstance(t, str):
            out.append(t)
    w(tree); return out


def render(tree):
    if isinstance(tree, tuple) and len(tree) == 2 and isinstance(tree[1], str):
        return tree[1]
    if isinstance(tree, tuple):
        return f"[{tree[0]} " + " ".join(render(k) for k in tree[1:]) + "]"
    return str(tree)


def main():
    store = Store(":memory:")
    build(store)
    gen = Generator(store)

    print("=" * 70)
    print("GENERATION SYSTEM (lattice-driven) — primes -> programs -> system")
    print("=" * 70)
    counts = {k: store.db.execute(f"SELECT count(*) FROM nodes WHERE kind='{k}'").fetchone()[0]
              for k in ("class", "word", "production", "frame", "inflect", "realize")}
    print(f"  programs in the generation system: {counts}\n")

    sents = []
    while len(sents) < 10:
        t = gen.generate()
        if not t: continue
        ws = leaves(t)
        gram = json.load(open("data/induced_grammar.json"))
        tags = ["det" if w in ("the","a") else "prep" if w=="to"
                else "verb" if w == ws[-1] and ("VP", ("verb", w)) else "noun" for w in ws]
        sents.append((render(t), ws))

    for tree, ws in sents:
        print(f"  {' '.join(ws):32}  tree: {tree}")

    print("\n  Every sentence: frame (lattice) -> proposition -> tree (productions")
    print("  from lattice) -> realized (rules from lattice). The generator is a")
    print("  generic interpreter; the knowledge is programs in the system tree.")
    print("\n  EXTENSIBILITY (system, not one-off): add a frame node ->")
    store.put_many([Node("program", "frame", "chase", type="predicate", agent="live",
                        substrate=[node_id("prime", "class", "ANIMATE")],
                        pathways={"sigs": [["ANIMATE", "ANIMATE"]]}),
                    Node("program", "inflect", "chase", type="morph", agent="live",
                        substrate=[node_id("prime", "category", "verb")],
                        pathways={"past": "chased"})])
    gen2 = Generator(store)
    for _ in range(60):
        t = gen2.generate()
        if t and "chased" in leaves(t):
            print(f"     new behavior, zero code change: {' '.join(leaves(t))}")
            break


if __name__ == "__main__":
    main()
