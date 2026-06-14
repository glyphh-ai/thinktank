"""
build.py — the engine: spin off agents to build a domain into the lattice.

Every domain is built the SAME way, bottom-up, and falls into one lattice:

    agents -> PRIMES   ->  agents -> PROGRAMS  ->  agent -> SYSTEM

Each agent is, at scale, an LLM call run in parallel; here they are
deterministic seed builders so the pipeline is runnable and exact. The point
is the ORCHESTRATION: whatever a swarm produces is validated by schema-on-write
and lands as typed nodes + substrate edges, so grammar, math, and (next)
reading all compose in the same graph.

This file builds two domains through one pipeline:
  * GRAMMAR  — category primes -> phrase-rule programs -> an english grammar
  * MATH     — digit/op primes -> number/expression programs -> arithmetic

From here "reading" is a program that traverses the lexicon + grammar SYSTEMS
(rules become data in the lattice, not hardcoded), and "comprehension" grounds
the result across towers (number-words -> the math tower). Those are the next
rungs; the seam is noted at the end.

Run:  python build.py
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Callable

from store import Store, Node, node_id


# ---------------------------------------------------------------------------
# An agent emits a batch of nodes for one layer of one domain. (LLM at scale.)
# ---------------------------------------------------------------------------
Agent = Callable[[], list[Node]]


@dataclass
class Domain:
    name: str
    prime_agents: list[Agent]
    program_agents: list[Agent]
    system_agent: Agent


class Lattice:
    """Runs domains through primes -> programs -> system, all into one store."""

    def __init__(self, store: Store):
        self.store = store

    def build(self, d: Domain) -> dict:
        # 1) spin prime-agents (a swarm; here run sequentially)
        primes = [n for a in d.prime_agents for n in a()]
        self.store.put_many(primes)
        # 2) spin program-agents — programs reference primes already in the store
        programs = [n for a in d.program_agents for n in a()]
        self.store.put_many(programs)
        # 3) the system — a collection of the domain's programs
        system = d.system_agent()
        self.store.put_many([system])
        return {"domain": d.name, "primes": len(primes),
                "programs": len(programs), "system": system.id}


# ===========================================================================
# DOMAIN: GRAMMAR  — the grammar system you asked for, as lattice nodes.
# ===========================================================================
GRAMMAR_CATEGORIES = ["det", "adj", "noun", "pron", "verb", "aux", "prep", "conj"]


def grammar_primes() -> list[Node]:
    # grammatical categories are fungible primes (untyped cells)
    return [Node("prime", "category", c, agent="agent:gram-primes",
                 pathways={"domain": "grammar"}) for c in GRAMMAR_CATEGORIES]


def cat_id(c: str) -> str:
    return node_id("prime", "category", c)


def grammar_programs() -> list[Node]:
    # phrase-structure rules are programs, typed 'phrase', composed of
    # categories (and, for PP/S, of OTHER rule-programs -> lattice depth).
    g = lambda exp: {"domain": "grammar", "expand": exp}
    np = Node("program", "rule", "NP", type="phrase",
              substrate=[cat_id("det"), cat_id("adj"), cat_id("noun")],
              agent="agent:gram-rules", pathways=g("det? adj* noun"))
    vp = Node("program", "rule", "VP", type="phrase",
              substrate=[cat_id("aux"), cat_id("verb")],
              agent="agent:gram-rules", pathways=g("aux* verb"))
    pp = Node("program", "rule", "PP", type="phrase",
              substrate=[cat_id("prep"), np.id],          # rule made of a rule
              agent="agent:gram-rules", pathways=g("prep NP"))
    s = Node("program", "rule", "S", type="phrase",
             substrate=[np.id, vp.id],
             agent="agent:gram-rules", pathways=g("NP VP"))
    return [np, vp, pp, s]


def grammar_system() -> Node:
    rules = [node_id("program", "rule", r) for r in ("NP", "VP", "PP", "S")]
    return Node("system", "grammar", "english", type="grammar",
                substrate=rules, agent="agent:gram-system",
                pathways={"domain": "grammar"})


GRAMMAR = Domain("grammar", [grammar_primes], [grammar_programs], grammar_system)


# ===========================================================================
# DOMAIN: MATH  — same pipeline, different primes.
# ===========================================================================
def math_primes() -> list[Node]:
    digits = [Node("prime", "digit", d, agent="agent:math-primes",
                   pathways={"domain": "math"}) for d in "0123456789"]
    ops = [Node("prime", "op", o, agent="agent:math-primes",
               pathways={"domain": "math"}) for o in "+-*/"]
    return digits + ops


def math_programs() -> list[Node]:
    integer = Node("program", "numtype", "integer", type="number",
                   substrate=[node_id("prime", "digit", d) for d in "0123456789"],
                   agent="agent:math-progs",
                   pathways={"domain": "math", "expand": "digit+"})
    expr = Node("program", "form", "expression", type="expression",
                substrate=[integer.id] + [node_id("prime", "op", o) for o in "+-*/"],
                agent="agent:math-progs",
                pathways={"domain": "math", "expand": "number (op number)*"})
    return [integer, expr]


def math_system() -> Node:
    return Node("system", "arithmetic", "core", type="math",
                substrate=[node_id("program", "numtype", "integer"),
                           node_id("program", "form", "expression")],
                agent="agent:math-system", pathways={"domain": "math"})


MATH = Domain("math", [math_primes], [math_programs], math_system)


# ===========================================================================
# RUN — build both domains into one lattice, then query it.
# ===========================================================================
def main():
    store = Store(":memory:")
    lat = Lattice(store)

    print("=" * 64)
    print("SPIN AGENTS — primes -> programs -> system, into one lattice")
    print("=" * 64)
    for d in (GRAMMAR, MATH):
        r = lat.build(d)
        print(f"  built {r['domain']:8}: {r['primes']:2} primes  "
              f"{r['programs']:2} programs  1 system")

    print("\n  unified lattice now holds:")
    for level in ("prime", "program", "system"):
        c = store.db.execute("SELECT count(*) FROM nodes WHERE level=?",
                             (level,)).fetchone()[0]
        print(f"     {level:8}: {c}")

    print("\n  GRAMMAR SYSTEM — its substrate is the phrase rules:")
    gsys = node_id("system", "grammar", "english")
    for rid in store.substrate_of(gsys):
        value, pathways = store.db.execute(
            "SELECT value, pathways FROM nodes WHERE id=?", (rid,)).fetchone()
        expand = json.loads(pathways).get("expand", "")
        print(f"     {value:3} -> {expand}")

    print("\n  LATTICE DEPTH — the PP rule is a program made of another program:")
    pp = node_id("program", "rule", "PP")
    print(f"     PP substrate -> {store.substrate_of(pp)}  (prep prime + NP rule)")

    print("\n  NEXT RUNGS (same lattice, same pipeline):")
    print("     reading       = a program that traverses the lexicon + grammar")
    print("                     SYSTEMS (rules are data here, not hardcoded).")
    print("     comprehension = grounding the parse across towers")
    print("                     (number-words -> the math tower, etc).")
    print("     -> spin the same primes->programs->system agents per domain;")
    print("        everything falls into this one graph.")


if __name__ == "__main__":
    main()
