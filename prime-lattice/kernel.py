"""
prime-lattice — a runnable sketch of the two-regime architecture.

The claim being tested
======================
Old-school GOFAI typed *everything* and got brittle. Deep nets type *nothing*
and get opaque. The bet here: TYPE AT THE LEVEL WHERE FUNCTION STABILIZES.

  - A cell is fungible: a cell is a cell. (ambiguous at the micro level)
  - A heart is typed: a heart is a heart -- same bio-material, but at a
    certain level of organisation a stable FUNCTION crystallises a TYPE.
  - Electricity (the signal) flows through every level the same way.

Mapped onto language:

  PRIME LAYER  (alphabet / sounds / symbols)   <- fungible "cells"
      * interchangeable, context-routed
      * ambiguity is CORRECT here and is NOT resolved by typing
        (the same 'c' is /k/ in cat and /s/ in city)

  PROGRAM LAYER  (words / sentences)           <- typed "organs"
      * typed by FUNCTION (part of speech / role)
      * crystallise out of recurring prime-assemblies (a promotion rule)
      * RETAIN their prime substrate -- a word can still be sounded out,
        the way a heart is still made of cells.

  SYSTEM LAYER   (lexicon / syntax / a language) <- typed "organ systems"
      * a collection of typed programs serving one function
      * itself a Node, so systems compose into systems-of-systems

The ladder is self-similar: neuron -> typed -> program -> system -> ...
each level is the same interface at a larger scale.

Both layers implement the SAME interface (a "neuron" is an interface, not an
implementation):

        receive(Signal) -> Signal

so a group of primes can stand in for a prime (the Composite pattern) and the
lattice grows without new machinery. A letter feeds MULTIPLE pathways
(orthographic -> word, phonetic -> sound): same atom, many parents => a
lattice, not a tree.

No model is in the loop at runtime. Everything below is deterministic, and the
one place determinism *strains* is flagged explicitly -- that flag is the whole
experiment.

Run:  python kernel.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol


# ---------------------------------------------------------------------------
# THE INTERFACE  ("neuron" = interface, not implementation)
# ---------------------------------------------------------------------------
@dataclass
class Signal:
    """What flows on a pathway. `payload` is the thing; `ctx` lets a junction
    route context-sensitively (e.g. the letter to the right of a 'c')."""
    payload: str
    ctx: dict = field(default_factory=dict)


class Node(Protocol):
    """Every unit -- a single letter or a whole sentence -- answers this.
    Substitutability (Liskov) is what lets composites masquerade as atoms."""
    name: str
    def receive(self, sig: Signal) -> Signal: ...


# ---------------------------------------------------------------------------
# PRIME LAYER  — fungible cells. Ambiguity lives here on purpose.
# ---------------------------------------------------------------------------
FRONT_VOWELS = set("iey")

# A few letters whose sound is genuinely context-free at this granularity.
SIMPLE = {
    "a": "/a/", "b": "/b/", "d": "/d/", "e": "/e/", "f": "/f/", "h": "/h/",
    "i": "/i/", "j": "/j/", "k": "/k/", "l": "/l/", "m": "/m/", "n": "/n/",
    "o": "/o/", "p": "/p/", "r": "/r/", "s": "/s/", "t": "/t/", "u": "/u/",
    "v": "/v/", "w": "/w/", "y": "/y/", "z": "/z/",
}


def route_sound(char: str, nxt: str | None) -> tuple[str, bool]:
    """A prime routed down the PHONETIC pathway. Context (`nxt`) selects the
    branch. Returns (phoneme, strained?) where `strained` marks a junction at
    which ONE RULE IS NOT ENOUGH -- the exact coordinate where deterministic
    routing needs help (a weight, a lookup, an exception list)."""
    c = char.lower()

    if c == "c":
        # Fully resolved by a rule: soft before front vowels, hard otherwise.
        return ("/s/", False) if (nxt and nxt.lower() in FRONT_VOWELS) else ("/k/", False)

    if c == "g":
        # The rule says soft /j/ before front vowels (gem, giant) ... but
        # 'get', 'give', 'gift' break it. The rule is NOT enough -> strained.
        if nxt and nxt.lower() in FRONT_VOWELS:
            return ("/j/", True)   # ambiguous: could be /g/ (get) — irreducible here
        return ("/g/", False)

    return SIMPLE.get(c, "?"), c not in SIMPLE


@dataclass
class Letter:
    """A fungible prime. It is 'just a letter' — interchangeable, untyped."""
    char: str
    name: str = ""

    def __post_init__(self):
        self.name = f"letter:{self.char}"

    def receive(self, sig: Signal) -> Signal:
        nxt = sig.ctx.get("next")
        phoneme, strained = route_sound(self.char, nxt)
        return Signal(phoneme, {"strained": strained})


# ---------------------------------------------------------------------------
# PROGRAM LAYER  — typed organs. Crystallised from prime-assemblies.
# ---------------------------------------------------------------------------
@dataclass
class Word:
    """A typed PROGRAM. Its type is its FUNCTION (part of speech). It is built
    from letter-primes and RETAINS them: `sound_out` re-runs the phonetic
    pathway over its own substrate, the way an organ is still made of cells."""
    spelling: str
    pos: str                     # the TYPE — only meaningful at this level
    name: str = ""

    def __post_init__(self):
        self.name = f"word:{self.spelling}"

    # same interface as a Letter -> a Word can stand wherever a Node can.
    def receive(self, sig: Signal) -> Signal:
        return Signal(self.spelling, {"pos": self.pos})

    def sound_out(self) -> list[tuple[str, str, bool]]:
        letters = [Letter(ch) for ch in self.spelling]
        out = []
        for i, let in enumerate(letters):
            nxt = self.spelling[i + 1] if i + 1 < len(self.spelling) else None
            s = let.receive(Signal(let.char, {"next": nxt}))
            out.append((let.char, s.payload, s.ctx["strained"]))
        return out


@dataclass
class Sentence:
    """A higher typed program, composed of word-programs and typed by the
    function each fills (subject / verb / object slot)."""
    words: list[Word]
    name: str = "sentence"

    def receive(self, sig: Signal) -> Signal:
        return Signal(" ".join(w.spelling for w in self.words),
                      {"shape": [w.pos for w in self.words]})


# ---------------------------------------------------------------------------
# SYSTEM LAYER  — a collection of typed programs that serve one function.
# A System is ITSELF a Node, so systems compose into systems: the same lattice
# machinery keeps going all the way up. (programs -> systems -> systems-of-...)
# ---------------------------------------------------------------------------
@dataclass
class System:
    role: str                # the system's function == its type at this level
    members: list            # list[Node]: words, sentences, or sub-systems
    name: str = ""

    def __post_init__(self):
        self.name = f"system:{self.role}"

    def receive(self, sig: Signal) -> Signal:
        return Signal(self.role, {"kind": "system",
                                  "members": [m.name for m in self.members]})


# ---------------------------------------------------------------------------
# THE CRYSTALLISATION LINE  — the promotion rule (the old "learning knot")
# ---------------------------------------------------------------------------
class Lexicon:
    """Where a fungible prime-assembly becomes a typed program.

    A spelling seen fewer than `promote_at` times is still 'just letters'.
    Once it recurs enough to pay for itself (a minimum-description-length /
    library-learning style criterion), it is REIFIED as a typed Word and from
    then on is used as a unit -- a heart, not a pile of cells."""

    def __init__(self, promote_at: int = 2):
        self.counts: dict[str, int] = {}
        self.programs: dict[str, Word] = {}
        self.promote_at = promote_at

    def observe(self, spelling: str, pos: str) -> bool:
        self.counts[spelling] = self.counts.get(spelling, 0) + 1
        if self.counts[spelling] >= self.promote_at and spelling not in self.programs:
            self.programs[spelling] = Word(spelling, pos)   # <-- crystallise
            return True
        return False

    def is_typed(self, spelling: str) -> bool:
        return spelling in self.programs


# ---------------------------------------------------------------------------
# DEMO  — tells the whole story, end to end, deterministically.
# ---------------------------------------------------------------------------
def banner(t): print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)


def main():
    banner("PRIME LAYER — fungible, context-routed, ambiguity allowed")
    strained_junctions = []
    for word in ["cat", "city", "gem", "get"]:
        chars = []
        for i, ch in enumerate(word):
            nxt = word[i + 1] if i + 1 < len(word) else None
            s = Letter(ch).receive(Signal(ch, {"next": nxt}))
            mark = "  <-- STRAINED (rule not enough)" if s.ctx["strained"] else ""
            if s.ctx["strained"]:
                strained_junctions.append((word, ch))
            chars.append(f"{ch}->{s.payload}{mark}")
        print(f"  {word:6} : " + "  ".join(chars))
    print("\n  Note: 'c' resolves by a clean rule (cat=/k/, city=/s/).")
    print("        'g' before i/e/y does NOT (gem=/j/ but get=/g/) -> flagged.")

    banner("CRYSTALLISATION LINE — primes -> typed programs (promotion rule)")
    lex = Lexicon(promote_at=2)
    stream = [("cat", "noun"), ("the", "det"), ("cat", "noun"),
              ("sat", "verb"), ("the", "det")]
    for spelling, pos in stream:
        promoted = lex.observe(spelling, pos)
        state = "TYPED program (organ)" if lex.is_typed(spelling) else "still letters (cells)"
        flag = "  *** crystallised ***" if promoted else ""
        print(f"  saw {spelling:5} ({pos:4}) -> {state}{flag}")

    banner("PROGRAM LAYER — typed by function, but substrate retained")
    cat = lex.programs["cat"]
    print(f"  word 'cat' is now a program typed: {cat.pos}")
    print(f"  ...yet still made of cells -> sound_out(): {cat.sound_out()}")

    # Build a sentence from the typed word-programs (compose organs).
    the, sat = Word("the", "det"), lex.programs.get("sat") or Word("sat", "verb")
    sentence = Sentence([the, cat, sat])
    out = sentence.receive(Signal(""))
    print(f"\n  sentence program: '{out.payload}'  typed shape: {out.ctx['shape']}")

    banner("SYSTEM LAYER — collections of typed programs, same interface")
    lexicon_sys = System("lexicon", list(lex.programs.values()))
    syntax_sys = System("syntax", [sentence])
    language_sys = System("language", [lexicon_sys, syntax_sys])  # system of systems
    for s in (lexicon_sys, syntax_sys, language_sys):
        info = s.receive(Signal(""))
        print(f"  {s.name:18} -> members: {info.ctx['members']}")
    print("\n  'language' is a System whose members are themselves Systems —")
    print("  the lattice recurses: neuron -> typed -> program -> system -> ...")

    banner("THE EMPIRICAL FINDING")
    print(f"  deterministic routing covered every junction EXCEPT {len(strained_junctions)}:")
    for w, ch in strained_junctions:
        print(f"     - '{ch}' in '{w}'  (needs a weight / lookup / exception)")
    print("\n  => ambiguity did NOT vanish; it localised to specific prime-level")
    print("     junctions. Type at the WORD level and those words are clean even")
    print("     where their letters were not. That is the two-regime claim,")
    print("     made falsifiable: probability (if any) belongs BELOW the line.")


if __name__ == "__main__":
    main()
