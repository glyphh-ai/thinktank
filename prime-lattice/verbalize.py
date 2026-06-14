"""
verbalize.py — fluent generation from the knowledge graph. NO LLM at runtime.

Reads the persistent lattice (entities, facts, relations) and produces flowing,
multi-sentence descriptions: groups facts by relation, opens with the category,
uses pronouns to avoid repetition, and combines same-relation facts into one
clause. Deterministic, auditable, ~zero marginal cost. No model in the loop.

Run:  python verbalize.py            # describe several entities
      python verbalize.py mars       # describe one
"""

from __future__ import annotations
import sys
from lattice import open_lattice
import wiki_brain


def join(xs):
    xs = list(dict.fromkeys(xs))                      # dedupe exact
    xs = [x for x in xs if not any(x != y and x in y for y in xs)]   # drop substrings
    if len(xs) == 1:
        return xs[0]
    if len(xs) == 2:
        return f"{xs[0]} and {xs[1]}"
    return ", ".join(xs[:-1]) + ", and " + xs[-1]


def an(w):
    return ("an " if w[:1].lower() in "aeiou" else "a ") + w


class Verbalizer:
    def __init__(self, store):
        self.brain = wiki_brain.Brain(store)

    def describe(self, e):
        fs = self.brain.facts_of(e)
        if not fs:
            return f"I haven't learned about {e}."
        by = {}
        for s, r, o in fs:
            by.setdefault(r, []).append(o)
        name = e.title()
        ref = "It"
        out = []

        # 1) opener: category (+ position if a planet)
        cat = join(by.get("is_a", ["thing"])[:2])
        opener = f"{name} is a {cat}"
        if "position_from_sun" in by:
            opener += f", the {by['position_from_sun'][0]} planet from the Sun"
        out.append(opener + ".")

        # 2) composition + color
        bits = []
        if "made_of" in by:
            bits.append(f"is made of {join(by['made_of'])}")
        if "color" in by:
            bits.append(f"is {join(by['color'])} in color")
        if bits:
            out.append(f"{ref} {' and '.join(bits)}.")

        # 3) relations to other things
        if "orbits" in by:
            out.append(f"{ref} orbits {join(by['orbits'])}.")
        if "located_in" in by:
            out.append(f"{ref} is located in {join(by['located_in'])}.")
        if "contains" in by:
            out.append(f"{ref} contains {join(by['contains'])}.")

        # 4) properties (combine has + a couple characteristics)
        props = list(by.get("has", []))
        if props:
            out.append(f"{ref} has {join(props[:4])}.")
        chars = [c for c in by.get("characteristic", []) if len(c.split()) <= 6][:2]
        for c in chars:
            out.append(f"{ref} is {c}.")

        if "named_after" in by:
            out.append(f"{name} is named after {by['named_after'][0]}.")

        return " ".join(out)


def main():
    store = open_lattice()
    v = Verbalizer(store)
    targets = sys.argv[1:] or ["mars", "comet", "gravity", "galaxy", "sun", "moon"]
    print("FLUENT GENERATION FROM THE GRAPH — no LLM, no GPU, ~$0 at runtime")
    print("=" * 68)
    for e in targets:
        print(f"\n{e.title()}:\n  {v.describe(e.lower())}")


if __name__ == "__main__":
    main()
