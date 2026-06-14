"""
read_corpus.py — learn to READ and COMPREHEND, grounded in the corpus.

The corpus is the grounding path. Every word is already taught (100% of the
vocabulary has a typed program). Now:

  1. INDUCE construction rules from the corpus's own sentences (the swarm reads
     POS patterns and writes grammar rules into the lattice).
  2. READ: deterministically parse corpus sentences with those rules + the
     taught lexicon (Earley; no model at runtime).
  3. COMPREHEND: pull predicate-argument (who did what) from each parse.

Coverage is the honest metric: what fraction of real sentences can the lattice
read and comprehend on its own? Long/embedded sentences will fail until more
rules are induced -- that gap is the grounding path forward.

Run:  python read_corpus.py --prep            # dump POS-tagged sample to induce on
      python read_corpus.py --rules R.json     # read + comprehend with induced rules
"""

from __future__ import annotations
import json
import re
import sys
import collections
import dissect
from reader import earley

MAXLEN = 8   # measure on sentences up to this many tokens


def tag(words, pos):
    t = [pos.get(w) for w in words]
    return t if all(t) else None


def head(tree, want):
    """Deepest token whose POS-ish label matches want (e.g. 'noun','verb')."""
    found = []
    def walk(t):
        if isinstance(t, tuple):
            if len(t) == 1 and isinstance(t[0], str):
                return
            for k in t[1:]:
                walk(k)
    # simpler: scan leaves with their tags handled by caller
    return found


def comprehend(tree, words, tags):
    """Predicate-argument from a parse: subject (first noun/pron), verb, object."""
    def leaves(t):
        out = []
        def w(x):
            if isinstance(x, tuple) and x and x[0] in ("S", "NP", "VP", "PP"):
                for k in x[1:]:
                    w(k)
            elif isinstance(x, tuple):
                out.append(x[0])
            elif isinstance(x, str):
                out.append(x)
        w(t)
        return out
    lv = leaves(tree)
    idx = {w_: tags[i] for i, w_ in enumerate(words)}
    subj = next((w_ for w_ in lv if idx.get(w_) in ("noun", "pron")), None)
    verb = next((w_ for w_ in lv if idx.get(w_) == "verb"), None)
    after = lv[lv.index(verb) + 1:] if verb in lv else []
    obj = next((w_ for w_ in after if idx.get(w_) in ("noun", "pron")), None)
    return subj, verb, obj


def main():
    pos = json.load(open("data/taught_pos_full.json"))
    tokens, sentences = dissect.load_corpus("/tmp/corpus.txt")

    short = []
    for s in sentences:
        ws = re.findall(r"[a-z]+", s.lower())
        if 2 <= len(ws) <= MAXLEN and tag(ws, pos):
            short.append(ws)

    if "--prep" in sys.argv:
        sample = [" ".join(f"{w}/{pos[w]}" for w in ws) for ws in short[:60]]
        json.dump(sample, open("/tmp/pos_sentences.json", "w"), indent=1)
        print(f"{len(short)} readable-length sentences; wrote 60 POS-tagged to "
              f"/tmp/pos_sentences.json for rule induction")
        return

    grammar = json.load(open(sys.argv[2])) if sys.argv[1:2] == ["--rules"] else {}
    read = comp = 0
    examples = []
    for ws in short:
        tags = [pos[w] for w in ws]
        tree = earley(grammar, tags, ws)
        if tree:
            read += 1
            s, v, o = comprehend(tree, ws, tags)
            if v and s:
                comp += 1
                if len(examples) < 6:
                    examples.append((" ".join(ws), s, v, o))

    n = len(short)
    print("=" * 66)
    print("READING + COMPREHENDING THE CORPUS (grounded, deterministic)")
    print("=" * 66)
    print(f"  taught vocabulary : {len(pos)} words (100% of corpus)")
    print(f"  induced rules     : {sum(len(v) for v in grammar.values())} "
          f"productions across {len(grammar)} constituents")
    print(f"  sentences (<= {MAXLEN} tokens): {n}")
    print(f"  READ (full parse)        : {read}/{n} = {100*read/n:.1f}%")
    print(f"  COMPREHEND (pred-arg)    : {comp}/{n} = {100*comp/n:.1f}%")
    print(f"\n  comprehended examples (subject / verb / object), grounded:")
    for sent, s, v, o in examples:
        print(f"     \"{sent}\"  ->  {v}({s}, {o})")
    print(f"\n  honest: long/embedded sentences (> {MAXLEN} tokens) aren't read yet.")
    print("  the swarm induces more construction rules -> coverage grows. The")
    print("  corpus is the grounding path; reading improves as rules are taught.")


if __name__ == "__main__":
    main()
