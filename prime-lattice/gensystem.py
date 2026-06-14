"""
gensystem.py — the generator. A small generic interpreter that READS the
generation system from the persistent lattice and walks a tree.

It hardcodes nothing: frames, noun classes, adjectives, productions and
realization rules all come from lattice.db (built by the swarm via ingest.py).
The bigger the system in the lattice, the richer the generation -- add more
programs, get more out, zero code change.

Run:  python ingest.py        # build/grow the generation system in the lattice
      python gensystem.py      # generate from whatever the lattice holds
"""

from __future__ import annotations
import collections
import json
import random
from lattice import open_lattice


class Generator:
    def __init__(self, store):
        self.by_class = collections.defaultdict(list)
        for w, p in store.db.execute("SELECT value, pathways FROM nodes WHERE kind='nclass'"):
            self.by_class[json.loads(p)["class"]].append(w)
        self.adj_by_class = collections.defaultdict(list)
        for w, p in store.db.execute("SELECT value, pathways FROM nodes WHERE kind='adjmod'"):
            for c in json.loads(p)["modifies"]:
                self.adj_by_class[c].append(w)
        self.frames = {v: json.loads(p) for v, p in store.db.execute(
            "SELECT value, pathways FROM nodes WHERE kind='frame'")}
        self.prods = set()
        for (p,) in store.db.execute("SELECT pathways FROM nodes WHERE kind='production'"):
            d = json.loads(p); self.prods.add((d["lhs"], tuple(d["rhs"])))

    def np(self, cls, adj=True):
        if cls not in self.by_class:
            return None
        noun = random.choice(self.by_class[cls])
        if (adj and random.random() < 0.45 and self.adj_by_class.get(cls)
                and ("NP", ("det", "adj", "noun")) in self.prods):
            return ("NP", "the", random.choice(self.adj_by_class[cls]), noun)
        return ("NP", "the", noun)

    def generate(self):
        v = random.choice(list(self.frames))
        fr = self.frames[v]
        subj_opts = [c for c in fr.get("subj", []) if c in self.by_class]
        if not subj_opts:
            return None
        subj = self.np(random.choice(subj_opts))
        vb = fr.get("past", v)
        obj = random.choice(fr.get("obj", ["NONE"]) or ["NONE"])
        if obj == "PLACE" and "PLACE" in self.by_class and ("VP", ("verb", "PP")) in self.prods:
            return ("S", subj, ("VP", ("verb", vb), ("PP", "to", self.np("PLACE", adj=False))))
        if obj in ("NONE",) or obj not in self.by_class:
            return ("S", subj, ("VP", ("verb", vb)))
        return ("S", subj, ("VP", ("verb", vb), self.np(obj)))


def words(tree):
    out = []
    def w(t):
        if isinstance(t, tuple) and t and isinstance(t[0], str) and t[0].isupper():
            for k in t[1:]: w(k)
        elif isinstance(t, tuple) and len(t) == 2:
            out.append(t[1])
        elif isinstance(t, str):
            out.append(t)
    w(tree); return out


def main():
    store = open_lattice()
    gen = Generator(store)
    n_fr = sum(len(v) for v in [gen.frames])
    print("=" * 70)
    print("GENERATION SYSTEM (persistent lattice) — bigger system, richer output")
    print("=" * 70)
    print(f"  reading from lattice.db: {len(gen.frames)} frames, "
          f"{sum(len(v) for v in gen.by_class.values())} class-tagged nouns, "
          f"{sum(len(v) for v in gen.adj_by_class.values())} adj-applications\n")
    seen = set()
    shown = 0
    while shown < 24:
        t = gen.generate()
        if not t:
            continue
        s = " ".join(words(t))
        if s in seen:
            continue
        seen.add(s)
        shown += 1
        print(f"  {s}")


if __name__ == "__main__":
    main()
