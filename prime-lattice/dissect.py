"""
dissect.py — take a massive corpus and dissect EVERY aspect into the lattice.

Start at each tower and build the programs that decompose the whole corpus:

  ALPHABET    : every letter            -> letter primes
  SOUNDS      : every word's phonemes   -> sound primes + pronunciation programs
  WORDS       : every token             -> word + spelling programs (-> letters)
  MORPHOLOGY  : affix decomposition     -> morpheme primes + morphology programs
  LEXICON/SEM : POS + meaning           -> taught by the SWARM (open class)
  GRAMMAR     : parse sentences         -> structure (uses taught POS)

The deterministic towers run over the FULL corpus. The open-class lexicon is
taught by real agents; because word frequency is Zipfian, teaching the top-N
words grounds most running text (the compounding again). Nothing is pre-given:
every aspect is built/taught into one schema-on-write lattice.

Run:  python dissect.py /tmp/corpus.txt
      python dissect.py /tmp/corpus.txt --taught taught.json
"""

from __future__ import annotations
import collections
import json
import re
import sys
import teach
from store import Store, Node, node_id

SUFFIXES = ["'s", "ing", "ed", "es", "ly", "er", "est", "s"]


def load_corpus(path):
    text = open(path, encoding="utf-8").read().lower()
    # strip Gutenberg header/footer if present
    m = re.search(r"\*\*\* start of .*?\*\*\*(.*)\*\*\* end of", text, re.S)
    if m:
        text = m.group(1)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    tokens = re.findall(r"[a-z]+", text)
    return tokens, sentences


def morph(word):
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            return [word[:-len(suf)], suf]
    return [word]


def dissect(store: Store, tokens, vocab):
    former = teach.WordFormer()
    # --- ALPHABET ---
    letters = sorted({c for w in vocab for c in w})
    store.put_many([Node("prime", "letter", c, agent="d:alpha") for c in letters])
    # --- SOUNDS + WORDS + MORPHOLOGY (per unique word) ---
    phon = {w: former.form(w).phonemes for w in vocab}
    sounds = sorted({p for ps in phon.values() for p in ps})
    morphemes = sorted({m for w in vocab for m in morph(w)})
    store.put_many([Node("prime", "sound", s, agent="d:phon") for s in sounds]
                   + [Node("prime", "morpheme", m, agent="d:morph") for m in morphemes])

    BATCH, buf = 4000, []
    def flush():
        nonlocal buf
        if buf: store.put_many(buf); buf = []
    for w in vocab:
        buf.append(Node("program", "spelling", w, type="form", agent="d:alpha",
                        substrate=[node_id("prime", "letter", c) for c in w]))
        buf.append(Node("program", "pronunciation", w, type="form", agent="d:phon",
                        substrate=[node_id("prime", "sound", s) for s in phon[w]] or
                                  [node_id("prime", "letter", w[0])]))
        ms = morph(w)
        buf.append(Node("program", "morphology", w, type="form", agent="d:morph",
                        substrate=[node_id("prime", "morpheme", m) for m in ms]))
        if len(buf) >= BATCH:
            flush()
    flush()
    return letters, sounds, morphemes, phon


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/corpus.txt"
    tokens, sentences = load_corpus(path)
    freq = collections.Counter(tokens)
    vocab = sorted(freq)

    store = Store(":memory:")
    letters, sounds, morphemes, phon = dissect(store, tokens, vocab)

    print("=" * 66)
    print("DISSECTED A MASSIVE CORPUS INTO THE LATTICE")
    print("=" * 66)
    print(f"  corpus     : {len(tokens):,} tokens, {len(vocab):,} unique words, "
          f"{len(sentences):,} sentences")
    print(f"\n  TOWERS BUILT (deterministic, full corpus):")
    print(f"     alphabet   : {len(letters)} letter primes")
    print(f"     sounds     : {len(sounds)} sound primes  (+{len(vocab):,} pronunciations)")
    print(f"     words      : {len(vocab):,} spelling programs")
    print(f"     morphology : {len(morphemes):,} morpheme primes (+{len(vocab):,} decompositions)")
    print(f"     lattice    : {store.count():,} nodes total")

    # Zipf coverage: teaching the top-N words grounds most running text
    print(f"\n  LEXICON tower is open-class -> taught by the swarm. Zipf coverage:")
    cum = 0
    for n in (50, 150, 500, 1000):
        cov = sum(c for _, c in freq.most_common(n))
        print(f"     teach top {n:>4} words -> grounds {100*cov/len(tokens):4.1f}% of all tokens")

    # write the top words for the swarm to teach (open-class only is fine here)
    top = [w for w, _ in freq.most_common(150)]
    json.dump(top, open("/tmp/teach_words.json", "w"))

    if len(sys.argv) > 3 and sys.argv[2] == "--taught":
        taught = json.load(open(sys.argv[3]))
        # teach POS into the lattice and measure running-text coverage
        store.put_many([Node("prime", "category", c, agent="seed")
                        for c in set(taught.values())])
        store.put_many([Node("program", "word", w, type=p, agent="a:teacher",
                            substrate=[node_id("prime", "category", p)])
                        for w, p in taught.items()])
        covered = sum(freq[w] for w in taught)
        print(f"\n  SWARM TAUGHT {len(taught)} words -> "
              f"{100*covered/len(tokens):.1f}% of running text now has a typed word-program")
        print(f"  lattice now: {store.count():,} nodes. Every aspect dissected/taught,")
        print(f"  nothing pre-given. Tail words get taught the same way, as met.")
    else:
        print(f"\n  -> wrote top-150 words to /tmp/teach_words.json for the swarm to teach.")


if __name__ == "__main__":
    main()
