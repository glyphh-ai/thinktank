"""
Strat 1 — THE LATTICE SCHEMA
============================
A fixed, named, weighted lattice grounded in the 65 NSM primes
(Wierzbicka/Goddard). Three levels, every node weighted:

    layer (weight) > segment (weight) > role (weight, filled per word)

Every role's allowed values are NSM primes (plus "none" = inapplicable).
The schema is FIXED — the same roles for every word — so codes are
comparable bit-for-bit and the judge can be certified. Haiku supplies,
per word, each role's (value, weight); segment and layer weights are
schema constants.

This file is pure data + validation. No API, no vectors.
"""

# ---- The 65 NSM primes, grouped as in Goddard's tables -----------------------
NSM_PRIMES = {
    "substantives":        ["i", "you", "someone", "people", "something", "body"],
    "relational":          ["kind", "part"],
    "determiners":         ["this", "the_same", "other"],
    "quantifiers":         ["one", "two", "much_many", "little_few", "some", "all"],
    "evaluators":          ["good", "bad"],
    "descriptors":         ["big", "small"],
    "mental_predicates":   ["think", "know", "want", "dont_want", "feel", "see", "hear"],
    "speech":              ["say", "words", "true"],
    "actions_events":      ["do", "happen", "move"],
    "existence_possession": ["be_somewhere", "there_is", "be_someone", "mine"],
    "life_death":          ["live", "die"],
    "time":                ["when_time", "now", "before", "after", "a_long_time",
                            "a_short_time", "for_some_time", "moment"],
    "space":               ["where_place", "here", "above", "below", "far", "near",
                            "side", "inside", "touch"],
    "logical":             ["not", "maybe", "can", "because", "if"],
    "intensifier":         ["very", "more"],
    "similarity":          ["like"],
}
ALL_PRIMES = [p for group in NSM_PRIMES.values() for p in group]
assert len(ALL_PRIMES) == 65, len(ALL_PRIMES)

# ---- The lattice -------------------------------------------------------------
# layer -> weight, segments; segment -> weight, roles; role -> allowed values.
# Role values are drawn from the prime groups above; "none" marks a role that
# does not apply to a word (its bound vector is skipped).
LATTICE = {
    "being": {
        "weight": 1.0,
        "segments": {
            "substance": {
                "weight": 1.0,
                "roles": {
                    "category":  ["someone", "people", "something", "body", "kind",
                                  "part", "words", "where_place", "when_time", "none"],
                    "animacy":   ["live", "die", "none"],
                    "quantity":  ["one", "two", "much_many", "little_few", "some",
                                  "all", "none"],
                },
            },
        },
    },
    "quality": {
        "weight": 0.9,
        "segments": {
            "value": {
                "weight": 1.0,
                "roles": {
                    "goodness":  ["good", "bad", "none"],
                    "size":      ["big", "small", "none"],
                    "intensity": ["very", "more", "none"],
                },
            },
        },
    },
    "mind": {
        "weight": 0.9,
        "segments": {
            "mental": {
                "weight": 1.0,
                "roles": {
                    "cognition": ["think", "know", "want", "dont_want", "feel",
                                  "see", "hear", "none"],
                    "speech":    ["say", "words", "true", "none"],
                },
            },
        },
    },
    "event": {
        "weight": 1.0,
        "segments": {
            "dynamics": {
                "weight": 1.0,
                "roles": {
                    "agency":    ["do", "happen", "move", "touch", "none"],
                    "existence": ["there_is", "be_somewhere", "be_someone", "mine",
                                  "none"],
                },
            },
        },
    },
    "context": {
        "weight": 0.7,
        "segments": {
            "situating": {
                "weight": 1.0,
                "roles": {
                    "temporal": ["now", "before", "after", "a_long_time",
                                 "a_short_time", "for_some_time", "moment",
                                 "when_time", "none"],
                    "spatial":  ["here", "above", "below", "far", "near", "side",
                                 "inside", "touch", "where_place", "none"],
                    "logical":  ["not", "maybe", "can", "because", "if", "like",
                                 "the_same", "other", "none"],
                },
            },
        },
    },
}

# Flat role paths: [("being", "substance", "category"), ...]
ROLE_PATHS = [
    (layer, seg, role)
    for layer, L in LATTICE.items()
    for seg, S in L["segments"].items()
    for role in S["roles"]
]
assert len(ROLE_PATHS) == 13, len(ROLE_PATHS)

# Every allowed value is an NSM prime (or "none").
for layer, seg, role in ROLE_PATHS:
    for v in LATTICE[layer]["segments"][seg]["roles"][role]:
        assert v == "none" or v in ALL_PRIMES, (layer, seg, role, v)


def validate_reading(reading: dict) -> list[str]:
    """Check one word's judged lattice reading; return a list of problems."""
    problems = []
    for layer, seg, role in ROLE_PATHS:
        key = f"{layer}.{seg}.{role}"
        entry = reading.get(key)
        if not isinstance(entry, dict):
            problems.append(f"missing role {key}")
            continue
        allowed = LATTICE[layer]["segments"][seg]["roles"][role]
        if entry.get("value") not in allowed:
            problems.append(f"{key}: bad value {entry.get('value')!r}")
        w = entry.get("weight")
        if not isinstance(w, (int, float)) or not (0.0 <= w <= 1.0):
            problems.append(f"{key}: bad weight {w!r}")
    return problems


if __name__ == "__main__":
    print(f"primes : {len(ALL_PRIMES)}")
    print(f"layers : {len(LATTICE)}  (weights: "
          + ", ".join(f"{k}={v['weight']}" for k, v in LATTICE.items()) + ")")
    print(f"roles  : {len(ROLE_PATHS)}")
    for lay, seg, role in ROLE_PATHS:
        vals = LATTICE[lay]["segments"][seg]["roles"][role]
        print(f"  {lay}.{seg}.{role:<10} <- {vals}")
