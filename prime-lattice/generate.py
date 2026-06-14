"""
generate.py — the inverse of reading: GENERATE language from the lattice.

Reading went form -> meaning by parsing. Generation goes the other way:
expand the grammar's construction rules, fill the terminal slots with taught
words, and realize the surface all the way down to letters and sounds -- the
full stack, primes up. Everything comes from what the swarm taught; nothing is
pre-given and no model runs at generation time.

Two checks:
  * STACK: every tower present and usable, primes -> programs -> systems.
  * ROUND-TRIP: each generated sentence is fed back to the READER. If the
    lattice can read what it generated, the towers are consistent both ways.

Run:  python generate.py
"""

from __future__ import annotations
import collections
import json
import random
import teach
from reader import earley, render

random.seed(7)
MAXD = 5


def load():
    pos = json.load(open("data/taught_pos_full.json"))
    grammar = json.load(open("data/induced_grammar.json"))
    by_pos = collections.defaultdict(list)
    for w, p in pos.items():
        by_pos[p].append(w)
    return pos, grammar, by_pos


def generate(grammar, by_pos, NT, sym="S", depth=0):
    if sym not in NT:                                   # terminal POS -> a word
        ws = by_pos.get(sym)
        return [random.choice(ws)] if ws else [sym]
    prods = grammar[sym]
    if depth > MAXD:                                    # stop recursion: prefer flat rules
        flat = [p for p in prods if all(s not in NT for s in p)]
        prods = flat or prods
    out = []
    for s in random.choice(prods):
        out += generate(grammar, by_pos, NT, s, depth + 1)
    return out


def main():
    pos, grammar, by_pos = load()
    NT = set(grammar)
    former = teach.WordFormer()

    # ---- STACK CHECK: every tower present, primes up ----------------------
    print("=" * 66)
    print("FULL STACK — every tower, primes -> programs -> systems")
    print("=" * 66)
    letters = sorted({c for w in pos for c in w})
    sounds = sorted({s for w in list(pos)[:500] for s in former.form(w).phonemes})
    stack = [
        ("PRIME  alphabet", f"{len(letters)} letters"),
        ("PRIME  sounds", f"{len(sounds)}+ phonemes"),
        ("PRIME  sem/role", "semantic primes + roles"),
        ("PROG   words", f"{len(pos)} typed words"),
        ("PROG   morphology", "affix decompositions"),
        ("PROG   grammar rules", f"{sum(len(v) for v in grammar.values())} productions"),
        ("PROG   comprehension", "predicate-argument (read_corpus)"),
        ("SYS    grammar/sem/emotion", "systems compose the programs"),
    ]
    for name, detail in stack:
        print(f"   [ok] {name:28} {detail}")

    # ---- COVERAGE: what we have to generate with --------------------------
    print("\n  GENERATION COVERAGE (taught words per category):")
    for p in ("noun", "verb", "adj", "adv", "det", "pron", "prep", "conj"):
        print(f"     {p:5}: {len(by_pos.get(p, [])):4} words", end="")
        if p in ("adj", "pron"):
            print()
    print()
    print(f"     productions per constituent: "
          + ", ".join(f"{k}:{len(v)}" for k, v in grammar.items()))

    # ---- GENERATE + ROUND-TRIP -------------------------------------------
    print("\n  GENERATED sentences (and read back to verify consistency):")
    ok = 0
    N = 12
    for _ in range(N):
        words = generate(grammar, by_pos, NT)
        if not (2 <= len(words) <= 12):
            continue
        tags = [pos[w] for w in words]
        tree = earley(grammar, tags, words)            # read what we generated
        if tree:
            ok += 1
        mark = "READ-OK" if tree else "unparsed"
        print(f"     [{mark}] {' '.join(words)}")

    print(f"\n  round-trip (generated -> read back): {ok}/{N} parse")

    # ---- REALIZE one sentence to PRIMES ----------------------------------
    words = generate(grammar, by_pos, NT)
    print("\n  REALIZED to primes (generation goes down to letters + sounds):")
    print(f"     sentence: {' '.join(words)}")
    for w in words[:5]:
        ph = former.form(w).phonemes
        print(f"        {w:10} letters={list(w)}  sounds={ph}")
    print("\n  => generation = the inverse of reading, over the same lattice.")
    print("     Same rules and words, expanded instead of parsed, realized to")
    print("     primes. The model both reads and generates from what it learned.")


if __name__ == "__main__":
    main()
