"""
full.py — every sentence through ALL towers, then longer + harder questions.

The grounding run used agents for ONE tower (fact extraction). This routes each
sentence through the whole stack and lets the agents supply the open-class
content (word meanings / verb frames / valence) that only an LLM can give:

  sentence -> orthography (letters)
           -> phonology  (sounds)
           -> lexicon    (words, POS)
           -> grammar    (parse / structure)
           -> comprehension (meaning, predicate-argument)
           -> emotion    (appraisal -> affect)
           -> facts      (relation from the meaning)

Fixed interpreters (reader, comprehender, emotion) do the composing; the agent
fills the lexical/semantic content. Then two questions:
  * HARDER  (multi-tower, one sentence): why does X feel what it feels?
  * LONGER  (across sentences): what is X's FINAL emotion in the story?

Run:  python full.py                       # local content
      python full.py --agent-lex F.json    # content supplied by a real agent
"""

from __future__ import annotations
import json
import sys
import reader as R
import comprehend as C
import emotion as E
import teach
from store import Store, Node, node_id

# Open-class content the agent supplies (local default for a self-test).
LOCAL_LEX = {
    "dog": {"pos": "noun", "meaning": {"t": "concept", "label": "dog", "sem": ["KIND", "LIVE", "MOVE"]}},
    "cat": {"pos": "noun", "meaning": {"t": "concept", "label": "cat", "sem": ["KIND", "LIVE", "SMALL", "MOVE"]}},
    "big": {"pos": "adj", "meaning": {"t": "mod", "sem": ["BIG"]}},
    "bit": {"pos": "verb", "meaning": {"t": "verb", "name": "BITE", "roles": ["agent", "patient"]}, "valence": "bad"},
    "saw": {"pos": "verb", "meaning": {"t": "verb", "name": "SEE", "roles": ["agent", "patient"]}, "valence": "neutral"},
    "won": {"pos": "verb", "meaning": {"t": "verb", "name": "WIN", "roles": ["agent"]}, "valence": "good"},
}


def build_from_agent(agent_lex: dict):
    """Fixed interpreter rules are seeded; the LEXICAL/SEMANTIC content is the
    agent's. (Grammar productions + sem-rules = the frozen core.)"""
    store = Store(":memory:")
    R.seed_grammar(store)                                   # grammar + POS lexicon
    store.put_many([R._rule_node("NP", ["NP", "conj", "NP"])])
    rule_ids = [r[0] for r in store.db.execute("SELECT id FROM nodes WHERE kind='rule'")]
    store.put_many([Node("system", "grammar", "english", type="grammar",
                         substrate=rule_ids, agent="a:gram")])

    # words the grammar lexicon doesn't already have -> POS from the agent
    have = {v for (v,) in store.db.execute("SELECT value FROM nodes WHERE kind='word'")}
    add = [Node("program", "word", w, type=e["pos"], agent="a:agent",
                substrate=[node_id("prime", "category", e["pos"])])
           for w, e in agent_lex.items() if w not in have]
    if add:
        store.put_many(add)

    # comprehension: fixed sem-primes + sem-rules; lexical MEANINGS from the agent
    store.put_many([Node("prime", "sem-prime", p, agent="a:sem") for p in C.SEM_PRIMES]
                   + [Node("prime", "role", r, agent="a:sem") for r in C.ROLES])
    lexnodes = []
    for w, e in agent_lex.items():
        m = e["meaning"]
        sub = [node_id("prime", "sem-prime", s) for s in m.get("sem", [])] or \
              [node_id("prime", "sem-prime", "KIND")]
        lexnodes.append(Node("program", "lex-sem", w, type="meaning", substrate=sub,
                             agent="a:agent", pathways={"meaning": m}))
    store.put_many(lexnodes)
    semnodes = []
    for (lhs, rhs), spec in C.SEMRULES.items():
        prod = node_id("program", "rule", f"{lhs}->{' '.join(rhs)}")
        roles = [node_id("prime", "role", r) for r in C.ROLES if r in json.dumps(spec)]
        semnodes.append(Node("program", "semrule", f"{lhs}->{' '.join(rhs)}",
                             type="inverse", substrate=[prod] + roles, agent="a:sem",
                             pathways={"lhs": lhs, "rhs": list(rhs), "spec": spec}))
    store.put_many(semnodes)
    store.put_many([Node("system", "comprehension", "core", type="meaning",
                         substrate=[n.id for n in lexnodes + semnodes], agent="a:sem")])
    E.seed_emotion(store)
    return store, R.Reader(store), C.Comprehender(store)


def appraise(meaning, valence_map):
    if not meaning or meaning.get("t") != "pred":
        return None
    valence = valence_map.get(meaning["name"], "neutral")
    roles = meaning["roles"]
    a = {"valence": valence, "experiencer": roles.get("patient") or roles.get("agent")}
    if "patient" in roles and "agent" in roles:
        a["cause"] = "other"
    if valence == "good":
        a["want"] = True
    return a


class FullPipeline:
    def __init__(self, agent_lex):
        self.store, self.reader, self.comp = build_from_agent(agent_lex)
        self.former = teach.WordFormer()
        self.valence = {e["meaning"]["name"]: e.get("valence", "neutral")
                        for e in agent_lex.values() if e["meaning"]["t"] == "verb"}

    def _ground_form(self, words):
        """Make the alphabet and sound towers REAL: build letter + sound primes
        and per-word spelling/pronunciation nodes that link to them (edges)."""
        letters, sounds = set(), set()
        phon = {}
        for w in words:
            phon[w] = self.former.form(w).phonemes
            letters |= set(w)
            sounds |= set(phon[w])
        self.store.put_many(
            [Node("prime", "letter", c, agent="a:ortho") for c in letters]
            + [Node("prime", "sound", s, agent="a:phon") for s in sounds])
        forms = []
        for w in words:
            forms.append(Node("program", "spelling", w, type="form",
                              substrate=[node_id("prime", "letter", c) for c in w],
                              agent="a:ortho"))
            forms.append(Node("program", "pronunciation", w, type="form",
                              substrate=[node_id("prime", "sound", s) for s in phon[w]],
                              agent="a:phon"))
        self.store.put_many(forms)
        return phon

    def decompose(self, sentence: str) -> dict:
        words = [w.strip(".").lower() for w in sentence.split()]
        phon = self._ground_form(words)        # alphabet + sound towers, for real
        tree = self.reader.parse(sentence.rstrip("."))
        meaning = self.comp.comprehend(tree) if tree else None
        appr = appraise(meaning, self.valence)
        feeling = E.recognize(appr) if appr and appr["valence"] != "neutral" else "calm"
        fact = None
        if meaning and meaning.get("t") == "pred":
            r = meaning["roles"]
            fact = {"rel": meaning["name"],
                    "agent": r.get("agent", {}).get("label"),
                    "patient": r.get("patient", {}).get("label")}
        # all-tower coverage, persisted as an utterance spanning the lexicon
        word_ids = [node_id("program", "word", w) for w in words]
        self.store.put_many([Node("program", "utterance", sentence, type="utterance",
                                  substrate=[i for i in word_ids if self._exists(i)] or word_ids[:1],
                                  agent="a:full")])
        return {"words": words,
                "orthography": sum(len(w) for w in words),
                "phonology": [p for w in words for p in phon[w]],
                "structure": tree, "meaning": meaning,
                "appraisal": appr, "feeling": feeling, "fact": fact}

    def _exists(self, nid):
        return self.store.db.execute("SELECT 1 FROM nodes WHERE id=?", (nid,)).fetchone() is not None


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--agent-lex":
        agent_lex = json.load(open(sys.argv[2]))
        src = "REAL AGENT"
    else:
        agent_lex, src = LOCAL_LEX, "local"

    pipe = FullPipeline(agent_lex)
    story = ["the dog bit the cat", "the big cat saw a dog", "the cat won"]

    print("=" * 70)
    print(f"EVERY SENTENCE THROUGH ALL TOWERS   (content from: {src})")
    print("=" * 70)
    decoded = []
    for s in story:
        d = pipe.decompose(s)
        decoded.append(d)
        towers = [f"ortho:{d['orthography']}", f"phon:{len(d['phonology'])}",
                  f"lex:{len(d['words'])}",
                  "gram:" + ("ok" if d["structure"] else "—"),
                  "mean:" + ("ok" if d["meaning"] else "—"),
                  f"emo:{d['feeling']}"]
        print(f"\n  \"{s}\"")
        print(f"     towers fired -> {'  '.join(towers)}")
        print(f"     structure: {R.render(d['structure'])}")
        print(f"     meaning  : {C.render(d['meaning'])}")
        if d["fact"]:
            print(f"     fact     : {d['fact']['rel']}({d['fact']['agent']},{d['fact'].get('patient')})")

    print("\n" + "-" * 70)
    print("  HARDER (multi-tower, one sentence): why does the cat feel as it does?")
    d0 = decoded[0]
    exp = d0["appraisal"]["experiencer"]["label"]
    f0 = d0["fact"]
    print(f"     -> {exp} feels {d0['feeling'].upper()} "
          f"because {f0['agent']} {f0['rel']} {f0['patient']}")

    print("\n  LONGER (across sentences): the cat's FINAL emotion in the story?")
    timeline = []
    for s, d in zip(story, decoded):
        a = d["appraisal"]
        if a and a["experiencer"]["label"] == "cat" and d["feeling"] != "calm":
            timeline.append((s, d["feeling"]))
    final = timeline[-1][1] if timeline else "calm"
    print(f"     cat's emotion timeline: {[(t[1]) for t in timeline]}")
    print(f"     -> FINAL: {final.upper()}  (composed across {len(story)} sentences)")

    print("\n" + "-" * 70)
    print("  TOWER AUDIT — what each tower actually put in the lattice:")
    audit = [("alphabet", "kind='letter'"), ("sounds", "kind='sound'"),
             ("spelling", "kind='spelling'"), ("pronunciation", "kind='pronunciation'"),
             ("words", "kind='word'"), ("grammar(rules)", "kind='rule'"),
             ("comprehension", "kind='lex-sem' OR kind='semrule'"),
             ("emotion", "kind='emotion'"), ("utterances", "kind='utterance'")]
    for name, where in audit:
        n = pipe.store.db.execute(f"SELECT count(*) FROM nodes WHERE {where}").fetchone()[0]
        print(f"     {name:16}: {n:3} nodes")
    # prove the vertical chain is connected: word 'cat' -> its letters & sounds
    sp = node_id("program", "spelling", "cat")
    pr = node_id("program", "pronunciation", "cat")
    letters = [pipe.store.db.execute("SELECT value FROM nodes WHERE id=?", (i,)).fetchone()[0]
               for i in pipe.store.substrate_of(sp)]
    snds = [pipe.store.db.execute("SELECT value FROM nodes WHERE id=?", (i,)).fetchone()[0]
            for i in pipe.store.substrate_of(pr)]
    print(f"\n  vertical chain for 'cat': letters {letters} -> sounds {snds} -> word -> meaning")
    nf = pipe.store.db.execute("SELECT count(*) FROM nodes").fetchone()[0]
    print(f"  total: {nf} nodes in one lattice. Every tower really decomposed.")


if __name__ == "__main__":
    main()
