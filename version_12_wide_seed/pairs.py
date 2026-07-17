"""version_12 seed inventory: wide named contrast pairs.

v11's diagnosis: every v10 axis was a difference of the same 61 prime
vectors, capping the whole bank at rank <= 60 — the barcode was blind to
~30% of human-relevant geometry. The fix is more SEED WORDS, not more pairs.

This file is a hand-curated inventory of English antonym/contrast pairs.
Every pair is self-naming (the bit for ("hot","cold") is called hot-cold).
Provenance: written from general lexical knowledge; NO benchmark ratings,
model vectors, or dataset files were consulted in choosing pairs. Seed words
may lexically overlap benchmark words (common antonyms inevitably do) — the
falsifier therefore also reports a no-overlap robustness split where every
benchmark pair containing a seed word is excluded.

The NSM prime pairs (v08 AXES) are folded in at the end so the old bits
remain available under their old names.
"""

PAIRS = [
    # --- adjectives: physical scales -----------------------------------------
    ("hot", "cold"), ("warm", "cool"), ("big", "small"), ("large", "tiny"),
    ("tall", "short"), ("wide", "narrow"), ("thick", "thin"), ("heavy", "light"),
    ("fast", "slow"), ("hard", "soft"), ("rough", "smooth"), ("sharp", "dull"),
    ("strong", "weak"), ("loud", "quiet"), ("bright", "dark"), ("clean", "dirty"),
    ("wet", "dry"), ("full", "empty"), ("open", "closed"), ("high", "low"),
    ("deep", "shallow"), ("tight", "loose"), ("fat", "thin"), ("straight", "curved"),
    ("sweet", "bitter"), ("sweet", "sour"), ("raw", "cooked"), ("bright", "dim"),
    ("smooth", "bumpy"), ("solid", "liquid"), ("long", "short"),
    # --- adjectives: evaluative / social -------------------------------------
    ("new", "old"), ("young", "old"), ("early", "late"), ("rich", "poor"),
    ("cheap", "expensive"), ("easy", "difficult"), ("simple", "complex"),
    ("safe", "dangerous"), ("happy", "sad"), ("calm", "angry"), ("kind", "cruel"),
    ("gentle", "harsh"), ("polite", "rude"), ("true", "false"), ("right", "wrong"),
    ("good", "evil"), ("beautiful", "ugly"), ("pretty", "ugly"), ("healthy", "sick"),
    ("alive", "dead"), ("awake", "asleep"), ("smart", "stupid"), ("wise", "foolish"),
    ("clever", "dumb"), ("sane", "crazy"), ("serious", "funny"), ("formal", "casual"),
    ("brave", "afraid"), ("proud", "ashamed"), ("guilty", "innocent"),
    ("legal", "illegal"), ("possible", "impossible"), ("useful", "useless"),
    ("major", "minor"), ("public", "private"), ("common", "rare"),
    ("normal", "strange"), ("modern", "ancient"), ("foreign", "domestic"),
    ("wild", "tame"), ("free", "trapped"), ("single", "married"), ("drunk", "sober"),
    ("famous", "unknown"), ("busy", "idle"), ("noisy", "silent"), ("lucky", "unlucky"),
    ("friendly", "hostile"), ("generous", "selfish"), ("humble", "arrogant"),
    ("patient", "impatient"), ("honest", "dishonest"), ("loyal", "treacherous"),
    ("tidy", "messy"), ("visible", "hidden"), ("permanent", "temporary"),
    ("ancient", "recent"), ("frequent", "rare"),
    # --- spatial / temporal ---------------------------------------------------
    ("near", "far"), ("close", "distant"), ("inner", "outer"), ("upper", "lower"),
    ("front", "back"), ("left", "right"), ("north", "south"), ("east", "west"),
    ("top", "bottom"), ("inside", "outside"), ("first", "last"), ("best", "worst"),
    ("more", "less"), ("many", "few"), ("all", "none"), ("always", "never"),
    ("often", "rarely"), ("day", "night"), ("morning", "evening"),
    ("summer", "winter"), ("spring", "autumn"), ("noon", "midnight"),
    ("sunrise", "sunset"), ("before", "after"), ("above", "below"),
    ("past", "future"), ("beginning", "end"), ("dawn", "dusk"),
    # --- verbs ----------------------------------------------------------------
    ("give", "take"), ("buy", "sell"), ("win", "lose"), ("love", "hate"),
    ("laugh", "cry"), ("rise", "fall"), ("push", "pull"), ("open", "shut"),
    ("come", "go"), ("arrive", "leave"), ("enter", "exit"), ("start", "stop"),
    ("begin", "finish"), ("build", "destroy"), ("create", "ruin"),
    ("live", "die"), ("remember", "forget"), ("find", "lose"), ("lend", "borrow"),
    ("teach", "learn"), ("ask", "answer"), ("speak", "listen"),
    ("shout", "whisper"), ("throw", "catch"), ("attack", "defend"),
    ("accept", "reject"), ("praise", "blame"), ("reward", "punish"),
    ("save", "spend"), ("increase", "decrease"), ("multiply", "divide"),
    ("float", "sink"), ("sleep", "wake"), ("marry", "divorce"), ("hire", "fire"),
    ("import", "export"), ("appear", "vanish"), ("expand", "shrink"),
    ("succeed", "fail"), ("laugh", "weep"), ("gather", "scatter"),
    ("freeze", "melt"), ("tie", "untie"), ("lock", "unlock"), ("bend", "straighten"),
    ("lift", "drop"), ("inhale", "exhale"), ("advance", "retreat"),
    ("obey", "rebel"), ("include", "exclude"), ("attract", "repel"),
    # --- nouns: roles / relations ---------------------------------------------
    ("king", "queen"), ("man", "woman"), ("boy", "girl"), ("father", "mother"),
    ("son", "daughter"), ("brother", "sister"), ("uncle", "aunt"),
    ("husband", "wife"), ("prince", "princess"), ("friend", "enemy"),
    ("hero", "villain"), ("angel", "devil"), ("heaven", "hell"),
    ("teacher", "student"), ("doctor", "patient"), ("buyer", "seller"),
    ("employer", "employee"), ("master", "servant"), ("host", "guest"),
    ("predator", "prey"), ("winner", "loser"), ("adult", "child"),
    ("expert", "beginner"), ("leader", "follower"), ("landlord", "tenant"),
    ("optimist", "pessimist"), ("ally", "rival"),
    # --- nouns: abstract ------------------------------------------------------
    ("war", "peace"), ("life", "death"), ("birth", "death"),
    ("question", "answer"), ("problem", "solution"), ("cause", "effect"),
    ("victory", "defeat"), ("success", "failure"), ("profit", "loss"),
    ("wealth", "poverty"), ("health", "disease"), ("joy", "sorrow"),
    ("pleasure", "pain"), ("courage", "fear"), ("hope", "despair"),
    ("truth", "lie"), ("fact", "fiction"), ("noise", "silence"),
    ("light", "shadow"), ("strength", "weakness"), ("order", "chaos"),
    ("entrance", "exit"), ("arrival", "departure"), ("minimum", "maximum"),
    ("majority", "minority"), ("presence", "absence"), ("reward", "penalty"),
    ("comedy", "tragedy"), ("angel", "demon"), ("sun", "moon"),
    ("fire", "ice"), ("land", "sea"), ("urban", "rural"), ("floor", "ceiling"),
    ("heat", "frost"), ("body", "soul"), ("work", "play"), ("logic", "emotion"),
]

# fold in the v08 NSM prime oppositions under their existing names
NSM_PAIRS = [
    ("good", "bad"), ("big", "small"), ("above", "below"), ("far", "near"),
    ("before", "after"), ("much", "little"), ("all", "some"), ("long", "moment"),
    ("live", "die"), ("someone", "something"), ("here", "there"),
    ("same", "other"), ("I", "you"),
]

ALL_PAIRS = PAIRS + [p for p in NSM_PAIRS if p not in PAIRS]
PAIR_NAMES = [f"{a}-{b}" for a, b in ALL_PAIRS]


def seed_words():
    return sorted({w for p in ALL_PAIRS for w in p})


def _validate():
    assert len(ALL_PAIRS) == len(set(ALL_PAIRS)), "duplicate pairs"
    assert len(set(PAIR_NAMES)) == len(PAIR_NAMES), "duplicate names"
    for a, b in ALL_PAIRS:
        assert a != b


_validate()

if __name__ == "__main__":
    print(f"{len(ALL_PAIRS)} named contrast pairs, {len(seed_words())} distinct "
          f"seed words -> rank ceiling ~{len(seed_words()) - 1} (v10: 60)")
