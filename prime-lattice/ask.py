"""
ask.py — read the PERSISTENT lattice and answer. No rebuild.

Proves persistence: a fresh process opens lattice.db (built by a previous
ingest run) and answers/generates from it directly. The knowledge survived the
run that created it.

Run:  python ask.py "What is Mars?"
      python ask.py --describe jupiter
"""

from __future__ import annotations
import sys
from lattice import open_lattice, stats
from wiki_brain import Brain


def main():
    store = open_lattice()
    s = stats(store)
    if s["total"] == 0:
        print("Lattice is empty — run `python ingest.py` first.")
        return
    brain = Brain(store)
    args = sys.argv[1:]
    print(f"[lattice.db: {s['total']} persisted nodes, no rebuild]\n")
    if args and args[0] == "--describe":
        print(brain.describe(args[1].lower()))
    else:
        q = " ".join(args) or "What is Mars?"
        print(f"Q: {q}\nA: {brain.ask(q)}")


if __name__ == "__main__":
    main()
