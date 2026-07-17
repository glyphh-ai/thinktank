"""version_08 axis inventory + PRE-REGISTERED polarity ground truth.

v07 killed "primes as dictionary atoms" and surfaced "primes as contrast
axes" (above~below cos 0.82, good~bad 0.69, before~after 0.60). This file
fixes, before any projection is computed:

  AXES        — 13 NSM-sanctioned prime oppositions. Axis direction is
                v(A) - v(B); the A-side is positive.
  POLARITY    — hand-coded ground truth for 3 axes (valence, size, animacy)
                over v07's 100 target words. Written BEFORE looking at any
                projection. AUC against these labels is H4.

Only clear-cut items are labeled; ambiguous ones (is a stone big? is fire
alive?) are deliberately left out rather than guessed.
"""

# --- the 13 prime contrast axes (A, B): positive side = A ---------------------
AXES = [
    ("good", "bad"),          # evaluation
    ("big", "small"),         # size
    ("above", "below"),       # vertical space
    ("far", "near"),          # distance
    ("before", "after"),      # temporal order
    ("much", "little"),       # quantity (mass)
    ("all", "some"),          # quantity (totality)
    ("long", "moment"),       # duration
    ("live", "die"),          # life
    ("someone", "something"), # animacy
    ("here", "there"),        # spatial deixis
    ("same", "other"),        # identity
    ("I", "you"),             # person deixis
]

AXIS_NAMES = [f"{a}-{b}" for a, b in AXES]

# --- H4 ground truth: expected sign of the projection -------------------------
# +1 -> the A side of the axis, -1 -> the B side.
POLARITY = {
    "good-bad": {
        "friend": +1, "mother": +1, "love": +1, "hope": +1, "peace": +1,
        "happy": +1, "help": +1, "laugh": +1, "dream": +1, "truth": +1,
        "warm": +1, "rich": +1, "strong": +1,
        "enemy": -1, "war": -1, "hate": -1, "fear": -1, "kill": -1,
        "fight": -1, "cry": -1, "angry": -1, "afraid": -1, "sad": -1,
        "poor": -1, "weak": -1, "lie": -1,
    },
    "big-small": {
        "mountain": +1, "sea": +1, "earth": +1, "sky": +1,
        "forest": +1, "city": +1, "sun": +1,
        "bird": -1, "cat": -1, "child": -1, "fish": -1,
    },
    "someone-something": {
        "dog": +1, "cat": +1, "bird": +1, "fish": +1, "horse": +1,
        "child": +1, "mother": +1, "father": +1, "friend": +1, "enemy": +1,
        "water": -1, "fire": -1, "stone": -1, "house": -1, "door": -1,
        "road": -1, "bread": -1, "money": -1, "ice": -1, "snow": -1,
    },
}


def _validate():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "version_07_prime_basis"))
    from concepts import PRIMES, TARGETS
    prime_set, target_set = set(PRIMES), set(TARGETS)
    axis_words = [w for pair in AXES for w in pair]
    assert len(axis_words) == len(set(axis_words)) == 26, "axis words must be 26 distinct primes"
    for w in axis_words:
        assert w in prime_set, f"axis word {w!r} is not a prime exponent"
    for name, labels in POLARITY.items():
        assert name in AXIS_NAMES, f"unknown axis {name!r}"
        for w in labels:
            assert w in target_set, f"labeled word {w!r} is not a v07 target"
        signs = list(labels.values())
        assert signs.count(+1) >= 4 and signs.count(-1) >= 4, f"{name}: need >=4 per side"


_validate()

if __name__ == "__main__":
    print(f"{len(AXES)} axes, polarity ground truth on "
          f"{sum(len(v) for v in POLARITY.values())} (axis, word) labels "
          f"across {len(POLARITY)} axes")
