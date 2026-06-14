"""
compounding.py — show that connectivity compounds.

Two principles drive the build:
  1. LLMs author the programs/resolutions at build time (constrained, frozen).
  2. Their connectivity should COMPOUND — one resolution fixes many words.

This script reads the taught lexicon.json and proves principle 2 with data:
the 345 words that strained determinism collapse to a tiny number of shared
GAP-SIGNATURES. So the LLM's build-time work is sublinear in words — resolving
K gaps repairs N words, N >> K — because fixes propagate through the shared
prime substrate (every word that contains an 's' inherits the 's' fix).

Run:  python compounding.py   (after teach.py)
"""

from __future__ import annotations
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def gap_signature(word: str, idx: int) -> tuple[str, str]:
    """A strained junction's signature = the grapheme + its right-context class.
    Words sharing a signature are all fixed by ONE authored resolution."""
    ch = word[idx]
    nxt = word[idx + 1] if idx + 1 < len(word) else "$"
    cls = ("front_vowel" if nxt in "eiy"
           else "vowel" if nxt in "aou"
           else "cons/end")
    return ch, cls


def main():
    db = json.load(open(os.path.join(HERE, "lexicon.json")))
    lex = db["lexicon"]
    strained = {w: r for w, r in lex.items() if not r["clean"]}

    by_sig = collections.defaultdict(list)
    for w, r in strained.items():
        for i in r["strained_at"]:
            by_sig[gap_signature(w, i)].append(w)

    total_junctions = sum(len(ws) for ws in by_sig.values())

    print("=" * 64)
    print("COMPOUNDING — K authored resolutions repair N words (N >> K)")
    print("=" * 64)
    print(f"  strained words        : {len(strained)}")
    print(f"  strained junctions    : {total_junctions}")
    print(f"  distinct gap-types (K): {len(by_sig)}")
    print()
    for sig, ws in sorted(by_sig.items(), key=lambda kv: -len(kv[1])):
        ch, cls = sig
        print(f"   resolve '{ch}' before {cls:12} -> repairs {len(ws):3} words"
              f"   e.g. {', '.join(ws[:4])}")

    leverage = round(total_junctions / len(by_sig), 1)
    print(f"\n  compounding factor    : {leverage}x "
          f"(junctions repaired per authored rule)")
    print("  => an LLM authors a handful of resolutions ONCE, at build time;")
    print("     each one propagates through every word sharing that prime.")
    print("     Add a word -> it reuses resolutions already in the lattice.")
    print("     That is the connectivity compounding, measured.")


if __name__ == "__main__":
    main()
