"""
emotion.py — affect is another tower, on the SAME primes.

NSM (Wierzbicka, "Emotions Across Languages and Cultures") explicates emotions
in semantic primes -- the same atoms as cognition. So emotion is not a special
module; it is one more tower in the lattice:

    new + shared primes  ->  emotion explications (programs)  ->  emotion system

Each emotion is a small cognitive scenario over primes (FEEL, GOOD, BAD, WANT,
BECAUSE, DO, HAPPEN...). It shares those primes with the semantics tower (multi-
inheritance), and it composes with comprehension: an appraisal of a comprehended
event maps onto the nearest emotion explication (appraisal theory / OCC, but the
content is NSM primes).

Run:  python emotion.py
"""

from __future__ import annotations
from store import Store, Node, node_id

# new + shared primes (GOOD/BAD/WANT/FEEL... are genuine NSM primes)
EMO_PRIMES = ["FEEL", "GOOD", "BAD", "WANT", "NOT", "DO", "HAPPEN",
              "BECAUSE", "KNOW", "CAN", "OTHER"]

# Each emotion: an NSM-style explication (clauses), the primes it uses, and an
# appraisal signature (the inverse: an event's appraisal -> this emotion).
EMOTIONS = {
    "anger": {
        "explication": ["something bad happened", "someone did this",
                        "I don't want this",
                        "I want to do something because of this",
                        "I feel something bad"],
        "primes": ["BAD", "HAPPEN", "OTHER", "DO", "NOT", "WANT", "BECAUSE", "FEEL"],
        "appraisal": {"valence": "bad", "cause": "other", "tendency": "act"},
    },
    "fear": {
        "explication": ["something bad can happen", "I don't know what to do",
                        "I feel something bad"],
        "primes": ["BAD", "CAN", "HAPPEN", "NOT", "KNOW", "DO", "FEEL"],
        "appraisal": {"valence": "bad", "certainty": "unknown", "tendency": "avoid"},
    },
    "joy": {
        "explication": ["something good happened", "I wanted this",
                        "I feel something good"],
        "primes": ["GOOD", "HAPPEN", "WANT", "FEEL"],
        "appraisal": {"valence": "good", "want": True},
    },
    "sadness": {
        "explication": ["something bad happened", "I can't do anything",
                        "I feel something bad"],
        "primes": ["BAD", "HAPPEN", "NOT", "CAN", "DO", "FEEL"],
        "appraisal": {"valence": "bad", "control": False, "tendency": "none"},
    },
}


def seed_emotion(store: Store):
    store.put_many([Node("prime", "sem-prime", p, agent="a:emo")
                    for p in EMO_PRIMES])
    progs = []
    for name, e in EMOTIONS.items():
        progs.append(Node("program", "emotion", name, type="affect",
                          substrate=[node_id("prime", "sem-prime", p) for p in e["primes"]],
                          agent="a:emo",
                          pathways={"explication": e["explication"],
                                    "appraisal": e["appraisal"]}))
    store.put_many(progs)
    store.put_many([Node("system", "emotion", "core", type="affect",
                         substrate=[p.id for p in progs], agent="a:emo")])


def recognize(appraisal: dict) -> str:
    """The inverse: appraisal of an event -> nearest emotion explication."""
    best, score = None, -1
    for name, e in EMOTIONS.items():
        s = sum(1 for k, v in e["appraisal"].items() if appraisal.get(k) == v)
        if s > score:
            best, score = name, s
    return best


def main():
    store = Store(":memory:")
    seed_emotion(store)

    print("=" * 66)
    print("EMOTION TOWER — affect reduces to the SAME primes as cognition")
    print("=" * 66)
    for name, e in EMOTIONS.items():
        print(f"\n  {name.upper():8} = " + " · ".join(e["explication"]))
        print(f"           primes: {' '.join(e['primes'])}")

    print("\n" + "-" * 66)
    print("  COMPOSITION WITH COMPREHENSION — appraise an event, feel an emotion")
    print("  " + "-" * 60)
    events = [
        ("the dog bit the cat", {"valence": "bad", "cause": "other", "tendency": "act"}),
        ("she won the game",    {"valence": "good", "want": True}),
        ("something moved in the dark", {"valence": "bad", "certainty": "unknown", "tendency": "avoid"}),
        ("the cat lost its friend", {"valence": "bad", "control": False, "tendency": "none"}),
    ]
    for text, appraisal in events:
        print(f"  {text:30} -> {recognize(appraisal)}")

    print("\n  => emotion shares FEEL/GOOD/BAD/WANT with the semantics tower")
    print("     (multi-inheritance), and the inverse (event -> emotion) is the")
    print("     same shape as comprehension. Same pipeline, affective tower.")


if __name__ == "__main__":
    main()
