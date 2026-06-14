"""
wiki_brain.py — grounding across systems on real Wikipedia data, then QA + generation.

Real agents extracted 127 fact-triples from 12 Wikipedia Solar-System intros.
This loads them into the lattice (entities + proposition + realization programs),
then:
  * ANSWERS questions by deterministic graph query (no model at runtime, no
    hallucination -- if it wasn't learned, it says so);
  * GENERATES descriptions from what it was seeded (the real test).

Run:  python wiki_brain.py
"""

from __future__ import annotations
import collections
import json
from store import Store, Node, node_id

# realization templates: a learned relation -> a surface sentence pattern
TEMPLATES = {
    "is_a": "{s} is a {o}", "position_from_sun": "{s} is the {o} planet from the Sun",
    "orbits": "{s} orbits {o}", "made_of": "{s} is made of {o}",
    "named_after": "{s} is named after {o}", "color": "{s} is {o} in color",
    "characteristic": "{s} is {o}", "has": "{s} has {o}",
    "distance": "{s} is at a distance of {o}", "located_in": "{s} is located in {o}",
    "larger_than": "{s} is larger than {o}", "satellite_of": "{s} is a satellite of {o}",
}


def build(store: Store, facts: dict):
    store.put_many([Node("prime", "sem-prime", "FACT", agent="w")])
    store.put_many([Node("prime", "entity", e, agent="w") for e in facts])
    fnodes = []
    for e, triples in facts.items():
        for i, t in enumerate(triples):
            fnodes.append(Node("program", "fact", f"{t['s']}|{t['r']}|{t['o']}",
                              type="proposition", agent="w",
                              substrate=[node_id("prime", "entity", t["s"])]
                                        if t["s"] in facts else [node_id("prime", "entity", e)],
                              pathways={"s": t["s"], "r": t["r"], "o": t["o"]}))
    store.put_many(fnodes)
    store.put_many([Node("program", "template", r, type="realize", agent="w",
                        substrate=[node_id("prime", "sem-prime", "FACT")],
                        pathways={"template": tmpl}) for r, tmpl in TEMPLATES.items()])
    store.put_many([Node("system", "knowledge", "solar", type="kg",
                        substrate=[n.id for n in fnodes], agent="w")])


class Brain:
    def __init__(self, store: Store):
        self.store = store
        self.T = [(json.loads(p)["s"], json.loads(p)["r"], json.loads(p)["o"])
                  for (p,) in store.db.execute("SELECT pathways FROM nodes WHERE kind='fact'")]
        self.templates = {v: json.loads(p)["template"] for v, p in store.db.execute(
            "SELECT value, pathways FROM nodes WHERE kind='template'")}
        self.entities = {e for (e,) in store.db.execute(
            "SELECT value FROM nodes WHERE kind='entity'")}

    def facts_of(self, e):
        return [(s, r, o) for s, r, o in self.T if s == e]

    def realize(self, s, r, o):
        return self.templates.get(r, "{s} {r} {o}").format(s=s.title(), r=r.replace("_", " "), o=o)

    # ---- GENERATION: describe an entity from its learned facts -------------
    def describe(self, e):
        fs = self.facts_of(e)
        if not fs:
            return f"I have not learned about {e}."
        order = ["is_a", "position_from_sun", "orbits", "made_of", "color",
                 "named_after", "has", "characteristic", "distance"]
        fs.sort(key=lambda t: order.index(t[1]) if t[1] in order else 99)
        return ". ".join(self.realize(s, r, o) for s, r, o in fs[:6]) + "."

    # ---- QA: deterministic graph query (no hallucination) -----------------
    def ask(self, q):
        ql = q.lower().rstrip("?")
        ent = next((e for e in self.entities if e in ql), None)
        # superlatives + ordinals first (they out-rank a generic "what is X")
        for adj in ("largest", "biggest", "hottest", "smallest", "coldest", "closest"):
            if adj in ql:
                hits = sorted({s for s, r, o in self.T if adj in o and s != "planet"})
                return f"{', '.join(hits)} ({adj})." if hits else f"I didn't learn which is {adj}."
        for ordn in ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"):
            if ordn in ql and "planet" in ql:
                hits = [s for s, r, o in self.T if r == "position_from_sun" and o == ordn]
                return f"The {ordn} planet from the Sun is {', '.join(hits)}." if hits else "Not learned."
        if any(w in ql for w in ("describe", "tell me about", "what is", "what's")) and ent:
            return self.describe(ent)
        if "orbit" in ql:
            tgt = "the sun" if "sun" in ql else ent
            hits = sorted({s for s, r, o in self.T if r == "orbits" and (tgt in o or o in (tgt or ""))})
            return ", ".join(hits) + " orbit " + (tgt or "?") if hits else "Nothing I learned orbits that."
        if "made of" in ql and ent:
            o = [o for s, r, o in self.facts_of(ent) if r == "made_of"]
            return f"{ent.title()} is made of {', '.join(o)}." if o else f"I didn't learn what {ent} is made of."
        for ordn in ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"):
            if ordn in ql and "planet" in ql:
                hits = [s for s, r, o in self.T if r == "position_from_sun" and o == ordn]
                return f"The {ordn} planet from the Sun is {', '.join(hits)}." if hits else "Not learned."
        for adj in ("largest", "biggest", "hottest", "smallest", "closest"):
            if adj in ql:
                hits = [s for s, r, o in self.T if adj in o]
                return f"{', '.join(hits)} ({adj})." if hits else "Not learned."
        if ent:
            return self.describe(ent)
        return "I can't answer that from what I learned."


def main():
    facts = json.load(open("data/wiki_facts.json"))
    store = Store(":memory:")
    build(store, facts)
    brain = Brain(store)

    print("=" * 70)
    print("GROUNDED ON WIKIPEDIA (12 articles, 127 facts) — QA + GENERATION")
    print("=" * 70)
    print(f"  lattice: {store.count()} nodes "
          f"({len(facts)} entities, {len(brain.T)} facts, {len(brain.templates)} templates)")

    print("\n  --- ASKING QUESTIONS (answered from the graph, no hallucination) ---")
    for q in ["What is Mars?", "What orbits the Sun?", "What is Jupiter made of?",
              "Which planet is fourth from the Sun?", "What is the largest planet?",
              "What is the hottest planet?", "Tell me about the Moon",
              "What is a unicorn?"]:
        print(f"\n  Q: {q}")
        print(f"  A: {brain.ask(q)}")

    print("\n  --- GENERATING from what it was seeded ---")
    for e in ["mars", "jupiter", "earth", "sun"]:
        print(f"\n  {e.title()}:")
        print(f"    {brain.describe(e)}")


if __name__ == "__main__":
    main()
