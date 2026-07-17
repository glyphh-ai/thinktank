"""version_07 concept inventory.

Three disjoint word sets:

  PRIMES   — single-word English exponents of the NSM semantic primes
             (Goddard & Wierzbicka 2014). The 65-prime inventory contains
             compounds ("DON'T WANT", "A LONG TIME", "FOR SOME TIME") and
             allolexic doublets ("SOMETHING~THING", "MUCH~MANY"); we take one
             single-word exponent per prime and fold/omit the compounds,
             giving 61 words. This is the candidate *privileged basis*.

  CONTROL_POOL — 122 high-frequency English words that are NOT primes and NOT
             targets. Random 61-word dictionaries drawn from this pool are the
             null hypothesis: "any 61 common words span meaning equally well."
             Frequency-matched on purpose — primes are almost all top-1000
             words, so beating a low-frequency control would prove nothing.

  TARGETS  — 100 mid-frequency content words (nouns/verbs/adjectives). NSM
             claims every one of these is explicable as a structured
             combination of primes. The linear-basis test asks a weaker
             question: is the *bundling* part of that claim visible in a tiny
             LM's activation geometry?
"""

# --- NSM primes: category -> single-word English exponents -------------------
PRIME_CATEGORIES = {
    "substantive":  ["I", "you", "someone", "people", "something", "body"],
    "relational":   ["kind", "part"],
    "determiner":   ["this", "same", "other"],
    "quantifier":   ["one", "two", "some", "all", "much", "little"],
    "evaluator":    ["good", "bad"],
    "descriptor":   ["big", "small"],
    "mental":       ["think", "know", "want", "feel", "see", "hear"],
    "speech":       ["say", "words", "true"],
    "action":       ["do", "happen", "move"],
    "existence":    ["be", "there", "mine"],
    "life":         ["live", "die"],
    "time":         ["when", "now", "before", "after", "moment", "long"],
    "space":        ["where", "here", "above", "below", "far", "near",
                     "side", "inside", "touch"],
    "logic":        ["not", "maybe", "can", "because", "if"],
    "intensity":    ["very", "more"],
    "similarity":   ["like"],
}

PRIMES = [w for ws in PRIME_CATEGORIES.values() for w in ws]
PRIME_CATEGORY = {w: c for c, ws in PRIME_CATEGORIES.items() for w in ws}

# --- frequency-matched control pool ------------------------------------------
# High-frequency words (function + light content), no primes, no targets.
CONTROL_POOL = [
    "the", "a", "an", "of", "to", "in", "for", "with", "on", "at",
    "by", "from", "up", "out", "down", "about", "into", "over", "under", "again",
    "then", "once", "during", "between", "through", "against", "among", "while", "until", "along",
    "but", "or", "so", "than", "too", "also", "just", "only", "even", "still",
    "yet", "ever", "never", "always", "often", "soon", "already", "almost", "quite", "rather",
    "he", "she", "they", "we", "it", "them", "us", "him", "her", "who",
    "what", "which", "whose", "each", "every", "both", "any", "no", "such", "own",
    "get", "make", "take", "come", "give", "put", "keep", "let", "seem", "turn",
    "show", "ask", "tell", "call", "try", "leave", "find", "bring", "hold", "become",
    "new", "old", "first", "last", "next", "high", "low", "right", "wrong", "sure",
    "real", "full", "whole", "certain", "several", "least", "less", "most", "enough", "few",
    "thing", "way", "man", "woman", "world", "day", "year", "hand", "place", "case",
    "fact", "group", "number", "point", "week", "night", "area", "name", "home", "side2",
]
# "side2" is a placeholder slot: "side" is a prime. Replace with a real word.
CONTROL_POOL[CONTROL_POOL.index("side2")] = "end"

# --- targets: 100 mid-frequency content words ---------------------------------
TARGETS = [
    # concrete nouns
    "water", "fire", "stone", "tree", "river", "mountain", "house", "door", "road", "bread",
    "dog", "cat", "bird", "fish", "horse", "child", "mother", "father", "friend", "enemy",
    "eye", "blood", "heart", "head", "mouth", "foot", "sun", "moon", "star", "rain",
    "wind", "snow", "ice", "earth", "sky", "sea", "island", "forest", "field", "city",
    # verbs
    "run", "walk", "jump", "swim", "fly", "eat", "drink", "sleep", "wake", "laugh",
    "cry", "sing", "dance", "fight", "kill", "help", "teach", "learn", "build", "break",
    "open", "close", "push", "pull", "throw", "catch", "cut", "burn", "grow", "fall",
    # adjectives
    "hot", "cold", "warm", "dry", "wet", "hard", "soft", "heavy", "light", "fast",
    "slow", "young", "strong", "weak", "rich", "poor", "happy", "sad", "angry", "afraid",
    # abstract nouns
    "love", "hate", "fear", "hope", "dream", "truth", "lie", "war", "peace", "money",
]

# --- carrier templates ---------------------------------------------------------
# Two frame families. USAGE frames put the word in running text (meaning-ish);
# MENTION frames quote it metalinguistically (form-ish). Both are identical
# across concepts, so centering across the vocabulary removes the frame itself.
# falsify.py checks the result survives on USAGE frames alone.
USAGE_TEMPLATES = [
    "It was all about {w}, nothing else.",
    "Everything here comes back to {w} in the end.",
    "She kept thinking about {w} all day.",
    "This story is mostly about {w}.",
    "There was something about {w} that mattered.",
    "My whole answer depends on {w}.",
]
MENTION_TEMPLATES = [
    'The word "{w}" is common in English.',
    'He wrote the word "{w}" on the board.',
    'She said the word "{w}" out loud.',
    'The meaning of "{w}" is easy to explain.',
    'Children learn the word "{w}" early.',
    'In this text, the word "{w}" appears twice.',
]
# Interleaved so even/odd split-halves mix both frame families.
TEMPLATES = [t for pair in zip(USAGE_TEMPLATES, MENTION_TEMPLATES) for t in pair]
FRAME_KIND = ["usage", "mention"] * len(USAGE_TEMPLATES)


def all_words():
    return PRIMES + CONTROL_POOL + TARGETS


def _validate():
    assert len(PRIMES) == 61, f"expected 61 prime exponents, got {len(PRIMES)}"
    assert len(TARGETS) == 100, f"expected 100 targets, got {len(TARGETS)}"
    assert len(CONTROL_POOL) >= 2 * len(PRIMES), "control pool too small for disjoint-ish resamples"
    p, c, t = set(PRIMES), set(CONTROL_POOL), set(TARGETS)
    assert len(p) == len(PRIMES) and len(c) == len(CONTROL_POOL) and len(t) == len(TARGETS), "duplicates within a set"
    assert not (p & c), f"prime/control overlap: {p & c}"
    assert not (p & t), f"prime/target overlap: {p & t}"
    assert not (c & t), f"control/target overlap: {c & t}"
    assert len(TEMPLATES) == 12 and len(FRAME_KIND) == 12


_validate()

if __name__ == "__main__":
    print(f"{len(PRIMES)} primes, {len(CONTROL_POOL)} control pool, {len(TARGETS)} targets")
    print(f"{len(TEMPLATES)} templates ({FRAME_KIND.count('usage')} usage / {FRAME_KIND.count('mention')} mention)")
