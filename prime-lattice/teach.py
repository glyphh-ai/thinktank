"""
teach.py — our first program: take LETTERS and their SOUNDS, and FORM WORDS.

This is the build-time "teaching" step. We seed three things, in order:

    1. THE ALPHABET      — 26 letter primes (fungible cells)
    2. THE SOUNDS        — a grapheme -> phoneme phonology, context-aware
    3. 1000 WORDS        — formed in the lattice by the WordFormer program

The WordFormer is the program that "knows how to take letters and sounds and
form words". Teaching produces a persisted lexicon (lexicon.json) — the
deterministic db the runtime reads. No model is in the loop.

Run:  python teach.py
Out:  prime-lattice/lexicon.json   + a coverage report
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# 1. THE ALPHABET — the prime set
# ===========================================================================
ALPHABET = [chr(c) for c in range(ord("a"), ord("z") + 1)]


# ===========================================================================
# 2. THE SOUNDS — phonology as a grapheme -> phoneme map.
#    Ambiguity is allowed and FLAGGED, never forced into a false type.
# ===========================================================================

# Digraphs: two letters, one sound. Scanned before single letters (greedy).
DIGRAPHS = {
    "th": "/θ/", "ch": "/tʃ/", "sh": "/ʃ/", "ph": "/f/", "wh": "/w/",
    "ck": "/k/", "ng": "/ŋ/", "qu": "/kw/", "ee": "/iː/", "oo": "/uː/",
    "ou": "/aʊ/", "ow": "/aʊ/", "ai": "/eɪ/", "ay": "/eɪ/", "oa": "/oʊ/",
    "ea": "/iː/", "igh": "/aɪ/",
}

# Single letters whose sound is ~context-free at this granularity.
LETTER_SOUND = {
    "a": "/æ/", "b": "/b/", "d": "/d/", "e": "/ɛ/", "f": "/f/", "h": "/h/",
    "i": "/ɪ/", "j": "/dʒ/", "k": "/k/", "l": "/l/", "m": "/m/", "n": "/n/",
    "o": "/ɒ/", "p": "/p/", "q": "/k/", "r": "/r/", "t": "/t/", "u": "/ʌ/",
    "v": "/v/", "w": "/w/", "x": "/ks/", "y": "/j/", "z": "/z/",
}

FRONT_VOWELS = set("eiy")


def sound_of(graphemes: str, idx: int) -> tuple[str, int, bool]:
    """Route one grapheme (1-3 letters) at position `idx` to a phoneme.
    Returns (phoneme, length_consumed, strained?).
    `strained=True` marks a junction where one rule is NOT enough — the exact
    coordinate where determinism needs a weight / lookup / exception."""
    three = graphemes[idx:idx + 3]
    two = graphemes[idx:idx + 2]
    ch = graphemes[idx]
    nxt = graphemes[idx + 1] if idx + 1 < len(graphemes) else None

    if three in DIGRAPHS:
        return DIGRAPHS[three], 3, False
    if two in DIGRAPHS:
        return DIGRAPHS[two], 2, False

    if ch == "c":                                   # clean rule: soft vs hard
        return ("/s/" if nxt in FRONT_VOWELS else "/k/"), 1, False
    if ch == "g":                                   # rule LEAKS: gem vs get
        return ("/dʒ/", 1, True) if nxt in FRONT_VOWELS else ("/g/", 1, False)

    if ch in LETTER_SOUND:
        return LETTER_SOUND[ch], 1, False
    return "?", 1, True                              # unknown -> strained


# ===========================================================================
# THE WORD-FORMER — the program that forms words from letters + sounds.
# ===========================================================================
@dataclass
class FormedWord:
    spelling: str
    phonemes: list[str]
    strained_at: list[int]          # letter indices where a rule wasn't enough
    syllables: int                  # a cheap structural type-feature

    @property
    def clean(self) -> bool:
        return not self.strained_at

    def as_record(self) -> dict:
        return {
            "spelling": self.spelling,
            "pron": "".join(self.phonemes),
            "phonemes": self.phonemes,
            "syllables": self.syllables,
            "clean": self.clean,
            "strained_at": self.strained_at,
        }


class WordFormer:
    """Knows how to take letters and their sounds and FORM a word in the
    lattice: it walks the spelling, greedily matches graphemes, routes each to
    a phoneme, and assembles a typed word-program that retains its substrate."""

    def form(self, spelling: str) -> FormedWord:
        phonemes, strained = [], []
        i = 0
        while i < len(spelling):
            ph, consumed, is_strained = sound_of(spelling, i)
            phonemes.append(ph)
            if is_strained:
                strained.append(i)
            i += consumed
        return FormedWord(spelling, phonemes, strained, _count_syllables(spelling))


_VOWELS = set("aeiouy")


def _count_syllables(word: str) -> int:
    """Rough syllable count — a structural feature the word is typed by."""
    n, prev_vowel = 0, False
    for ch in word:
        v = ch in _VOWELS
        if v and not prev_vowel:
            n += 1
        prev_vowel = v
    if word.endswith("e") and n > 1:                 # silent final e
        n -= 1
    return max(1, n)


# ===========================================================================
# TEACH — seed alphabet + sounds, then form 1000 words into the lexicon db.
# ===========================================================================
def load_corpus() -> list[str]:
    with open(os.path.join(HERE, "data", "words_1000.txt")) as f:
        return [w.strip() for w in f if w.strip()]


def teach() -> dict:
    former = WordFormer()
    corpus = load_corpus()

    lexicon = {}
    for w in corpus:
        lexicon[w] = former.form(w).as_record()

    clean = sum(1 for r in lexicon.values() if r["clean"])
    strained_words = [w for w, r in lexicon.items() if not r["clean"]]

    db = {
        "alphabet": ALPHABET,
        "sounds": {"digraphs": DIGRAPHS, "letters": LETTER_SOUND},
        "lexicon": lexicon,
        "stats": {
            "words": len(lexicon),
            "clean": clean,
            "strained": len(strained_words),
            "coverage_pct": round(100 * clean / len(lexicon), 1),
        },
    }
    out = os.path.join(HERE, "lexicon.json")
    with open(out, "w") as f:
        json.dump(db, f, ensure_ascii=False, indent=1)
    return db, strained_words, out


def main():
    db, strained_words, out = teach()
    s = db["stats"]
    print("=" * 64)
    print("TAUGHT: alphabet -> sounds -> words")
    print("=" * 64)
    print(f"  alphabet primes : {len(db['alphabet'])}")
    print(f"  digraph sounds  : {len(db['sounds']['digraphs'])}")
    print(f"  words formed    : {s['words']}")
    print(f"  formed cleanly  : {s['clean']}  ({s['coverage_pct']}%)")
    print(f"  hit a strained  : {s['strained']} junctions need a weight/lookup")
    print(f"  saved db        : {os.path.relpath(out, HERE)}")

    print("\n  sample words the program formed:")
    for w in ["the", "cat", "city", "knight", "school", "though"]:
        r = db["lexicon"].get(w)
        if r:
            tag = "" if r["clean"] else "  <-- strained"
            print(f"     {w:8} -> {r['pron']:14} syl={r['syllables']}{tag}")

    print(f"\n  first 12 words that strained determinism ({s['strained']} total):")
    print("     " + ", ".join(strained_words[:12]))
    print("\n  => these are the exact coordinates where letters+sounds alone")
    print("     under-determine the word; everything else formed with no model.")


if __name__ == "__main__":
    main()
