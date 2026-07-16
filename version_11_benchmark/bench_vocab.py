"""v11 extraction vocabulary: every unique word of SimLex-999 + WordSim-353 +
MEN (1819 words), plus a valence-decile-stratified sample of Warriner words
(120 per decile, seed 0) to widen the norms tests. Resolved list committed as
bench_wordlist.txt. Selection uses only the datasets — no model signal.
"""
import csv
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
WORDLIST = os.path.join(HERE, "bench_wordlist.txt")
PER_DECILE = 120


def benchmark_pairs(name):
    pairs = []
    for line in open(os.path.join(DATA, name)):
        a, b, s = line.split()
        pairs.append((a.lower(), b.lower(), float(s)))
    return pairs


def warriner():
    out = {}
    with open(os.path.join(DATA, "warriner.csv")) as f:
        for row in csv.DictReader(f):
            w = row["Word"].lower()
            if w.isalpha():
                out[w] = float(row["V.Mean.Sum"])
    return out


def build():
    words = set()
    for name in ("simlex.txt", "ws353.txt", "men.txt"):
        for a, b, _ in benchmark_pairs(name):
            words.add(a); words.add(b)
    assert all(w.isalpha() for w in words)

    val = warriner()
    ranked = sorted(val, key=val.get)
    rng = random.Random(0)
    extra = []
    n = len(ranked)
    for d in range(10):
        decile = [w for w in ranked[d * n // 10:(d + 1) * n // 10] if w not in words]
        extra += rng.sample(decile, min(PER_DECILE, len(decile)))
    final = sorted(words) + sorted(extra)
    with open(WORDLIST, "w") as f:
        f.write("\n".join(final) + "\n")
    return final


def load_vocab():
    return [w.strip() for w in open(WORDLIST) if w.strip()]


if __name__ == "__main__":
    v = build()
    print(f"wrote {WORDLIST}: {len(v)} words "
          f"({len(v) - 10 * PER_DECILE} benchmark + stratified Warriner sample)")
