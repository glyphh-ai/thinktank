"""
sentence.py — the next program up: form PARTS OF A SENTENCE from words.

The ladder so far:
    letters + sounds  ->  WORDS            (teach.py)
    words             ->  CONSTITUENTS     (this file: NP / VP / PP)
    constituents      ->  SENTENCE

A Constituent is a "part of a sentence" — a phrase typed by its grammatical
function. Same Node interface as a Word, so it slots into the lattice and
nests (a PP contains an NP). Each constituent RETAINS its substrate: it still
knows its words, which still know their letters and sounds.

Typing words by function splits cleanly along a real linguistic seam, and that
seam tells the swarm exactly what to do:

  * CLOSED-class words (det, prep, conj, pronoun, aux) are a FINITE set — we
    seed them directly, exhaustively. They are almost primes themselves.
  * OPEN-class words (noun, verb, adjective) are unbounded — this is what the
    swarm of agents TEACHES, one slice of the dictionary at a time.

Run:  python sentence.py
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ===========================================================================
# CLOSED-CLASS function words — finite, so we seed them exhaustively here.
# ===========================================================================
FUNCTION_WORDS = {
    **{w: "det" for w in "the a an this that these those my your his her its "
       "our their some any no every each all both".split()},
    **{w: "pron" for w in "i you he she it we they me him her them us who what "
       "which".split()},
    **{w: "prep" for w in "on in at by for to of with from into over under "
       "about as between through".split()},
    **{w: "conj" for w in "and or but so yet nor because although if while "
       "when".split()},
    **{w: "aux" for w in "is are was were be been being am do does did have "
       "has had will would can could should may might must".split()},
}

# OPEN-class stand-ins for the demo. In the real system each of these is taught
# by a swarm agent (and stored alongside the word's spelling/sound substrate).
DEMO_OPEN = {"cat": "noun", "dog": "noun", "mat": "noun", "child": "noun",
             "sat": "verb", "ran": "verb", "sleeps": "verb", "big": "adj"}


def pos_of(word: str) -> str:
    """A word's grammatical TYPE. Closed class is known; open class is taught."""
    if word in FUNCTION_WORDS:
        return FUNCTION_WORDS[word]
    return DEMO_OPEN.get(word, "open")          # 'open' = awaiting swarm teaching


# ===========================================================================
# THE CONSTITUENT — a part of a sentence, typed by function. A Node, so it
# nests and composes exactly like a word.
# ===========================================================================
@dataclass
class Constituent:
    label: str                  # NP / VP / PP — the type at this level
    members: list               # list[Node]: words or sub-constituents
    name: str = ""

    def __post_init__(self):
        self.name = f"{self.label}:" + " ".join(
            m.text if isinstance(m, Tok) else m.label for m in self.members)

    def receive(self, sig):     # same interface as a Word
        return None

    def render(self) -> str:
        inner = " ".join(m.text if isinstance(m, Tok) else m.render()
                         for m in self.members)
        return f"[{self.label} {inner}]"


@dataclass
class Tok:
    """A minimal word-token carrying its grammatical type. (In the full system
    this is the Word program from kernel.py, substrate and all.)"""
    text: str
    pos: str


# ===========================================================================
# THE PROGRAM — form parts of a sentence from a stream of typed words.
# A tiny right-linear constituent grammar; the point is the LATTICE move, not
# coverage: words crystallise into typed phrases, phrases into a sentence.
# ===========================================================================
class ConstituentFormer:
    def form(self, words: list[str]) -> Constituent:
        toks = [Tok(w, pos_of(w)) for w in words]
        parts: list = []
        i = 0
        while i < len(toks):
            t = toks[i]
            if t.pos == "prep":                                  # PP -> prep NP
                np, i = self._noun_phrase(toks, i + 1)
                parts.append(Constituent("PP", [t] + ([np] if np else [])))
            elif t.pos in ("det", "noun", "pron", "adj"):        # NP
                np, i = self._noun_phrase(toks, i)
                parts.append(np)
            elif t.pos in ("verb", "aux"):                       # VP -> aux* verb
                vp_members = [t]
                i += 1
                while i < len(toks) and toks[i].pos in ("aux", "verb"):
                    vp_members.append(toks[i]); i += 1
                parts.append(Constituent("VP", vp_members))
            else:
                parts.append(Constituent("?", [t])); i += 1
        return Constituent("S", parts)

    def _noun_phrase(self, toks, i):
        members = []
        while i < len(toks) and toks[i].pos in ("det", "adj"):
            members.append(toks[i]); i += 1
        if i < len(toks) and toks[i].pos in ("noun", "pron", "open"):
            members.append(toks[i]); i += 1
        return (Constituent("NP", members) if members else None), i


def main():
    former = ConstituentFormer()
    print("=" * 64)
    print("NEXT PROGRAM UP — words -> parts of a sentence (constituents)")
    print("=" * 64)
    for sentence in [
        "the big cat sat on the mat",
        "she ran to the child",
        "the dog sleeps",
    ]:
        s = former.form(sentence.split())
        print(f"\n  {sentence}")
        print(f"    {s.render()}")

    print("\n  Each [NP]/[VP]/[PP] is a typed part of a sentence — a program at")
    print("  the next level, same interface as a word, retaining its substrate.")
    print("\n  Closed-class words (det/prep/aux...) are seeded here, finite.")
    print("  Open-class words (noun/verb/adj) are what the SWARM teaches: one")
    print("  agent per dictionary slice, each storing a word's POS next to its")
    print("  spelling and sound — and those types compound into every phrase.")


if __name__ == "__main__":
    main()
