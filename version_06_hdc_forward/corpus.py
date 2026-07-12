"""
corpus.py — the joint ingestion pass: ONE write path -> BOTH sides of the lattice.
=================================================================================
Per the design doc (adaL/docs/glyph-ada-training-loop.md), every fact is emitted
as two isomorphic views that share provenance and cannot drift:

  symbolic side (Ada) : each content WORD tagged with its Ada slot (entity.name,
                        spatial.location, relational.object, ...).  -> role ids
  vector side (Glyph) : the same fact's (role, filler) pairs bound + bundled into
                        one bipolar HDC cortex.                     -> hdc.Encoder

Word-level (see thinktank/version_05_lattice): the HDC forward-pass guidance
checks similarity at WORD boundaries, so the corpus is word-tokenized.

Bigger than v05's toy: 3 domains (people / geography / chemistry), procedurally
filled templates, real cross-domain role ambiguity (e.g. a place name is
spatial.location after "lives in" but spatial.origin after "is from"; an element
name is entity.name in one template, relational.object in another).

Run:  python corpus.py    # builds the corpus, prints stats, verifies both sides align
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from hdc import Encoder, bind, raw_cosine

# ── Ada universal schema (vendored from glyphh-ai/adaL, Apache-2.0) ──────────
UNIVERSAL_SCHEMA = {
    "entity":       ["name", "kind", "subkind"],
    "perceptual":   ["color", "size", "shape", "texture", "sound", "smell", "taste", "temperature"],
    "spatial":      ["location", "origin", "direction"],
    "temporal":     ["time", "duration", "age", "era", "frequency"],
    "relational":   ["subject", "predicate", "object", "possessor", "agent", "patient", "instrument"],
    "quantitative": ["count", "magnitude", "unit", "ratio"],
    "epistemic":    ["source", "certainty", "modality"],
}
ALL_ROLES = [f"{L}.{r}" for L, rs in UNIVERSAL_SCHEMA.items() for r in rs]
ROLES = ["O"] + ALL_ROLES                 # "O" = function/filler word
ROLE_ID = {r: i for i, r in enumerate(ROLES)}
N_ROLE = len(ROLES)
SEP = "<sep>"
UNK = "<unk>"

# ── Domain lexicons (procedurally combined -> a big corpus) ──────────────────
PEOPLE = [
    dict(name=n, loc=l, origin=o, job=j, pet=p, color=c, age=a, count=k)
    for n, l, o, j, p, c, a, k in [
        ("carol", "austin", "ohio", "engineer", "parrot", "green", "thirty", "two"),
        ("dave", "denver", "texas", "teacher", "dog", "brown", "forty", "one"),
        ("erin", "boston", "maine", "doctor", "cat", "black", "fifty", "three"),
        ("frank", "miami", "cuba", "chef", "fish", "gold", "sixty", "four"),
        ("gina", "seattle", "oregon", "pilot", "rabbit", "white", "twenty", "five"),
        ("hank", "chicago", "iowa", "lawyer", "snake", "red", "thirty", "one"),
        ("ivy", "portland", "idaho", "nurse", "bird", "blue", "forty", "two"),
        ("jack", "dallas", "kansas", "farmer", "horse", "grey", "fifty", "six"),
        ("kate", "phoenix", "arizona", "artist", "turtle", "green", "twenty", "one"),
        ("leo", "atlanta", "georgia", "banker", "mouse", "brown", "sixty", "two"),
        ("mona", "tucson", "nevada", "writer", "hawk", "amber", "thirty", "three"),
        ("nate", "fresno", "utah", "sailor", "goat", "silver", "forty", "one"),
        ("olga", "reno", "montana", "welder", "crow", "violet", "fifty", "four"),
        ("pete", "tampa", "florida", "baker", "lizard", "orange", "sixty", "two"),
        ("quinn", "boise", "wyoming", "ranger", "otter", "teal", "twenty", "five"),
    ]
]
# geography: place -> region (shares place names with PEOPLE.loc/origin -> ambiguity)
GEO = [
    ("austin", "texas", "river", "south"), ("denver", "colorado", "mountain", "west"),
    ("boston", "massachusetts", "harbor", "east"), ("miami", "florida", "beach", "south"),
    ("seattle", "washington", "sound", "west"), ("chicago", "illinois", "lake", "north"),
    ("portland", "oregon", "bridge", "west"), ("dallas", "texas", "plain", "south"),
    ("phoenix", "arizona", "desert", "west"), ("atlanta", "georgia", "forest", "east"),
]
# chemistry: element facts (element name is entity.name here, relational.object elsewhere)
CHEM = [
    ("helium", "gas", "light", "two", "cold"), ("carbon", "solid", "black", "six", "warm"),
    ("oxygen", "gas", "clear", "eight", "cold"), ("iron", "solid", "grey", "twenty", "warm"),
    ("neon", "gas", "bright", "ten", "cold"), ("sodium", "solid", "silver", "eleven", "warm"),
    ("argon", "gas", "clear", "eighteen", "cold"), ("copper", "solid", "orange", "twenty", "warm"),
    ("silver", "solid", "white", "forty", "warm"), ("gold", "solid", "gold", "sixty", "warm"),
]


def _people_facts(e) -> List[List[Tuple[str, str]]]:
    return [
        [(e["name"], "entity.name"), ("lives", "relational.predicate"), ("in", "O"), (e["loc"], "spatial.location")],
        [(e["name"], "entity.name"), ("is", "O"), ("from", "O"), (e["origin"], "spatial.origin")],
        [(e["name"], "entity.name"), ("works", "relational.predicate"), ("as", "O"), ("a", "O"), (e["job"], "relational.object")],
        [(e["name"], "entity.name"), ("owns", "relational.predicate"), ("a", "O"), (e["pet"], "relational.object")],
        [("the", "O"), (e["pet"], "entity.name"), ("is", "O"), (e["color"], "perceptual.color")],
        [(e["name"], "entity.name"), ("is", "O"), (e["age"], "temporal.age"), ("years", "quantitative.unit"), ("old", "O")],
        [(e["name"], "entity.name"), ("has", "relational.predicate"), (e["count"], "quantitative.count"), ("pets", "relational.object")],
    ]


def _geo_facts(g) -> List[List[Tuple[str, str]]]:
    place, region, feature, direction = g
    return [
        [(place, "entity.name"), ("is", "O"), ("in", "O"), (region, "spatial.location")],
        [(place, "entity.name"), ("has", "relational.predicate"), ("a", "O"), (feature, "relational.object")],
        [(place, "entity.name"), ("lies", "relational.predicate"), ("to", "O"), ("the", "O"), (direction, "spatial.direction")],
    ]


def _chem_facts(c) -> List[List[Tuple[str, str]]]:
    el, kind, color, number, temp = c
    return [
        [(el, "entity.name"), ("is", "O"), ("a", "O"), (kind, "entity.kind")],
        [(el, "entity.name"), ("looks", "relational.predicate"), (color, "perceptual.color")],
        [(el, "entity.name"), ("has", "relational.predicate"), (number, "quantitative.count"), ("protons", "relational.object")],
        [(el, "entity.name"), ("feels", "relational.predicate"), (temp, "perceptual.temperature")],
    ]


def all_fact_templates() -> List[List[Tuple[str, str]]]:
    facts: List[List[Tuple[str, str]]] = []
    for e in PEOPLE:
        facts += _people_facts(e)
    for g in GEO:
        facts += _geo_facts(g)
    for c in CHEM:
        facts += _chem_facts(c)
    return facts


def fact_role_fillers(fact: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Collapse a tagged word list into canonical (role, filler) pairs (drop 'O')."""
    out: List[Tuple[str, str]] = []
    cur_role, cur_words = None, []
    for w, r in fact:
        if r == "O":
            continue
        if r == cur_role:
            cur_words.append(w)
        else:
            if cur_role is not None:
                out.append((cur_role, " ".join(cur_words)))
            cur_role, cur_words = r, [w]
    if cur_role is not None:
        out.append((cur_role, " ".join(cur_words)))
    return out


def build_corpus(repeat: int = 60, seed: int = 0, vocab_cap: int = 4000,
                 vector_dim: int = 10000, encoder_seed: int = 42):
    """
    Emit both sides of the lattice from one write path.

    Returns dict with:
      ids   (N,)   token ids           }  symbolic
      rids  (N,)   Ada role ids         }  side
      spans list of (start,end,fact_key) per emitted fact (into ids/rids)
      vocab, stoi
      cortex   {fact_key: bipolar cortex np.int8}   } vector
      fillers  {fact_key: [(role, filler)]}          } side  (Ada slots recovered)
      enc      the hdc.Encoder used
    """
    rng = np.random.default_rng(seed)
    templates = all_fact_templates()

    # vocab cap first pass
    wc = Counter(w for f in templates for w, _ in f)
    keep = {w for w, _ in wc.most_common(vocab_cap)}

    def norm(w):
        return w if w in keep else UNK

    toks: List[str] = []
    roles: List[str] = []
    spans: List[Tuple[int, int, str]] = []
    enc = Encoder(vector_dim=vector_dim, seed=encoder_seed)
    cortex: Dict[str, np.ndarray] = {}
    fillers: Dict[str, List[Tuple[str, str]]] = {}

    order = list(range(len(templates)))
    for _ in range(repeat):
        rng.shuffle(order)
        for ti in order:
            fact = templates[ti]
            rf = fact_role_fillers(fact)
            fact_key = f"t{ti}"
            if fact_key not in cortex:
                cortex[fact_key] = enc.encode(rf)   # vector side (canonical, unaffected by <unk>)
                fillers[fact_key] = rf              # symbolic side
            start = len(toks)
            for w, r in fact:
                toks.append(norm(w))
                roles.append(r)
            spans.append((start, len(toks), fact_key))
            toks.append(SEP)
            roles.append("O")

    vocab = sorted(set(toks))
    stoi = {w: i for i, w in enumerate(vocab)}
    ids = np.array([stoi[w] for w in toks], dtype=np.int64)
    rids = np.array([ROLE_ID[r] for r in roles], dtype=np.int64)
    return dict(ids=ids, rids=rids, spans=spans, vocab=vocab, stoi=stoi,
                cortex=cortex, fillers=fillers, enc=enc)


if __name__ == "__main__":
    c = build_corpus(repeat=60)
    ids, rids, vocab = c["ids"], c["rids"], c["vocab"]
    templates = all_fact_templates()

    # ambiguous words: same surface word, >1 distinct Ada role across the corpus
    from collections import defaultdict
    word_roles = defaultdict(set)
    for f in templates:
        for w, r in f:
            if r != "O":
                word_roles[w].add(r)
    ambiguous = {w for w, rs in word_roles.items() if len(rs) > 1}
    roles_used = sorted({ROLES[r] for r in set(rids.tolist())} - {"O"})

    print("=== corpus (both sides of the lattice, one write path) ===")
    print(f"distinct fact templates : {len(templates)}")
    print(f"emitted tokens          : {len(ids):,}")
    print(f"vocab                   : {len(vocab)}")
    print(f"Ada roles in play       : {len(roles_used)}  {roles_used}")
    print(f"ambiguous surface words : {len(ambiguous)}  e.g. "
          + ", ".join(f"{w}={sorted(word_roles[w])}" for w in list(ambiguous)[:4]))
    print(f"HDC cortices (facts)    : {len(c['cortex'])}  dim={c['enc'].D}")

    # verify the two sides agree: unbinding each fact's roles from its cortex
    # recovers the RIGHT filler (symbolic side) far above a random filler.
    enc = c["enc"]
    all_fillers = list({f for rf in c["fillers"].values() for _, f in rf})
    rng = np.random.default_rng(1)
    hits, tot, margin = 0, 0, []
    for key, rf in c["fillers"].items():
        cx = c["cortex"][key]
        for role, filler in rf:
            guess = bind(cx, enc.role_vec(role))
            s_true = raw_cosine(guess, enc.filler_vec(filler))
            rand_f = all_fillers[rng.integers(len(all_fillers))]
            s_rand = raw_cosine(guess, enc.filler_vec(rand_f))
            hits += int(s_true > s_rand)
            margin.append(s_true - s_rand)
            tot += 1
    print(f"\nisomorphy check: unbind(cortex, role) -> correct filler "
          f"beats random {hits}/{tot} = {hits/tot:.1%}  (mean margin {np.mean(margin):+.3f})")
    assert hits / tot > 0.95, "the two lattice views disagree — provenance broken"
    print("both sides agree — symbolic slots are recoverable from the vector cortex. OK")
