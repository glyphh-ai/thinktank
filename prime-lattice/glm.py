"""
glm.py — a Generated Language Model. Chat with the lattice.

Not a neural net: every answer comes from programs the swarm built into a
persistent lattice (knowledge graph, frames, synonyms, hypernyms, inference,
generation). It resolves aliases (relations), remembers what you're talking
about (memory), answers from grounded facts (no hallucination), reasons over
is-a (inference), and can generate (the generation system).

Run interactively:   python glm.py
Or pipe a script:    printf 'what is a comet?\nwhat is it made of?\n' | python glm.py
"""

from __future__ import annotations
import json
import re
import sys
from lattice import open_lattice
import wiki_brain
import gensystem


class GLM:
    def __init__(self):
        self.store = open_lattice()
        self.brain = wiki_brain.Brain(self.store)          # knowledge + QA + generation
        self.gen = gensystem.Generator(self.store)         # generation system
        self.alias = {}
        for _, p in self.store.db.execute("SELECT value, pathways FROM nodes WHERE kind='synonym'"):
            d = json.loads(p)
            # only multi-word aliases ("red planet"->mars); single words over-rewrite
            if d["alias"] and d["canonical"] and " " in d["alias"]:
                self.alias[d["alias"]] = d["canonical"]
        self.hyper = {v: json.loads(p)["hyper"] for v, p in self.store.db.execute(
            "SELECT value, pathways FROM nodes WHERE kind='hypernym'")}
        self.focus = None      # memory: last entity discussed
        self.history = []

    # --- inference: is x a y? (is_a facts + hypernym chain) ----------------
    def isa(self, x, y):
        cats = {o for s, r, o in self.brain.facts_of(x) if r in ("is_a", "part_of")}
        if x in self.hyper:
            cats.add(self.hyper[x])
        hit = any(y == c or y in c.split() or c in y for c in cats)
        return hit, cats

    def respond(self, text):
        raw = text.strip()
        t = raw.lower().rstrip("?.!")
        if self.focus:                                     # memory: resolve pronouns
            t = re.sub(r"\b(it|its|that|this|they|them)\b", self.focus, t)
        for a, c in self.alias.items():                    # relations: resolve aliases
            if a in t:
                t = re.sub(rf"\b{re.escape(a)}\b", c, t)
        r = self._answer(t, raw)
        self.history.append((raw, r))
        return r

    def _answer(self, t, raw):
        if re.match(r"(hi|hello|hey|greetings)\b", t):
            return ("Hello — I'm a generated language model. My knowledge is "
                    "programs in a lattice, grounded on Wikipedia. Ask me about "
                    "planets, the sun, comets, galaxies, gravity — or say "
                    "'generate a sentence'.")
        if "who are you" in t or "what are you" in t:
            return ("A generated language model: my knowledge lives as programs "
                    "the swarm built into a lattice, not in neural weights. I "
                    "answer from grounded facts and say so when I don't know.")
        if t.startswith(("generate", "say something", "make a sentence", "talk")):
            tree = self.gen.generate()
            return " ".join(gensystem.words(tree)).capitalize() + "." if tree else "..."
        # is X a Y  (inference)
        m = re.match(r"is (?:an? |the )?(\w+) (?:an? |the )?(\w[\w ]*)$", t)
        if m:
            x, y = m.group(1), m.group(2).strip()
            if x in self.brain.entities or x in self.hyper:
                self.focus = x
                hit, cats = self.isa(x, y)
                if hit:
                    return f"Yes — a {x} is a {y}."
                cat = self.hyper.get(x) or (next(iter(cats)) if cats else None)
                return f"No — a {x} is a {cat}." if cat else f"I'm not sure what a {x} is."
        # entity question -> remember it, answer from the graph
        ent = next((e for e in self.brain.entities if re.search(rf"\b{re.escape(e)}\b", t)), None)
        if ent:
            self.focus = ent
        ans = self.brain.ask(t)
        if ans and not ans.startswith("I can't"):
            return ans
        if self.focus:
            return self.brain.describe(self.focus)
        return ("I haven't learned that yet — try a planet, the sun, a comet, "
                "a galaxy, gravity, or 'generate a sentence'.")


def main():
    glm = GLM()
    s = glm.store.count()
    print(f"== Generated Language Model ==  [{s} programs in the lattice, no neural weights]")
    print("type a message (ctrl-D to exit)\n")
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit"):
            break
        print(f"glm> {glm.respond(line)}\n")


if __name__ == "__main__":
    main()
