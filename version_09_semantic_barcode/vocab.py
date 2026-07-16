"""version_09 evaluation vocabulary + PRE-REGISTERED composition probes.

EVAL vocabulary: the top ~1000 alphabetic words (length >= 3) of the
google-10000 US-English frequency list, minus the 61 prime exponents (axes
and eval words must be disjoint), plus every word a composition probe
mentions. The resolved list is committed as wordlist.txt — the source of
truth for reproducibility; this module regenerates/validates it.

COMPOSITION PROBES: fixed BEFORE any barcode was computed. Each probe is
(a, b, expected-any-of): the bundle of a's and b's barcodes should rank at
least one expected word highly among the whole vocabulary. Probes were
filtered only by "is the word in the 10k list" — never by looking at
activations or codes.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORDLIST = os.path.join(HERE, "wordlist.txt")
N_TOP = 1000

# (a, b, [expected...]) — bundle(code(a), code(b)) should retrieve an expected word
PROBES = [
    ("water", "cold", ["ice", "snow"]),
    ("big", "water", ["sea", "ocean", "lake"]),
    ("bad", "person", ["enemy", "criminal"]),
    ("good", "person", ["friend", "hero"]),
    ("baby", "dog", ["puppy"]),
    ("money", "place", ["bank", "market"]),
    ("sick", "place", ["hospital", "clinic"]),
    ("book", "place", ["library", "school"]),
    ("young", "man", ["boy"]),
    ("young", "woman", ["girl"]),
    ("water", "sky", ["rain", "cloud", "storm"]),
    ("cold", "season", ["winter"]),
    ("royal", "man", ["king", "prince"]),
    ("war", "person", ["soldier", "army"]),
    ("school", "person", ["teacher", "student"]),
]


def probe_words():
    ws = set()
    for a, b, exp in PROBES:
        ws.add(a); ws.add(b); ws.update(exp)
    return ws


def build_wordlist(g10k_path):
    import sys
    sys.path.insert(0, os.path.join(HERE, "..", "version_07_prime_basis"))
    from concepts import PRIMES
    prime_set = set(PRIMES) | {"i"}  # prime exponent "I" is lowercase in the list
    raw = [w.strip() for w in open(g10k_path)]
    keep, seen = [], set()
    for w in raw:
        if len(keep) >= N_TOP:
            break
        if len(w) >= 3 and w.isalpha() and w not in prime_set and w not in seen:
            keep.append(w); seen.add(w)
    for w in sorted(probe_words()):
        if w not in seen:
            assert w in raw, f"probe word {w!r} not in the 10k list"
            keep.append(w); seen.add(w)
    with open(WORDLIST, "w") as f:
        f.write("\n".join(keep) + "\n")
    return keep


def load_vocab():
    words = [w.strip() for w in open(WORDLIST) if w.strip()]
    missing = probe_words() - set(words)
    assert not missing, f"probe words missing from wordlist: {missing}"
    return words


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:  # regenerate: python vocab.py path/to/g10k.txt
        words = build_wordlist(sys.argv[1])
        print(f"wrote {WORDLIST}: {len(words)} words")
    v = load_vocab()
    print(f"vocab: {len(v)} words, {len(PROBES)} probes, all probe words present")
