"""
HUMAN QUESTIONS — plain-language riddles a person would ask
===========================================================
Written by hand from general knowledge, WITHOUT looking at any glyph, so the
questions cannot leak answers. Answers are words that have glyphs.

The brain hears each question's words (minus a short list of filler words:
"it", "the", "you", ...); any word the brain has never encountered is
dropped and counted (coverage). It answers by flow.py's thinking: one stream
per word, flowing through the four regions, coincidence where they meet,
stopping on its own.

Contestants:
  lookup     count question words that appear in each word's own glyph
  brain      flow.py, untrained
  brain+     flow.py with pathway gains learned on the synthetic riddles
             (it never saw these questions)

Run:  python human_questions.py
"""

import re

import numpy as np

import flow as FL
import reason as R
import think as T

QUESTIONS = [
    ("doctor", "who works in a hospital and helps sick people get better"),
    ("hospital", "a building where sick people go to be treated by doctors and nurses"),
    ("teacher", "a person who helps students learn at school"),
    ("school", "a place where children go to learn with teachers and books"),
    ("book", "you read it, it has pages and a story"),
    ("clock", "it tells you the time, it has hands and it ticks"),
    ("coin", "small round metal money you keep in your pocket"),
    ("wall", "the solid side of a room, made of brick"),
    ("tower", "a very tall narrow building"),
    ("music", "songs and sounds you listen to and dance to"),
    ("mask", "you wear it over your face to hide who you are"),
    ("spy", "a secret agent who watches people and steals information"),
    ("enemy", "someone who hates you and fights against you"),
    ("friend", "someone you like and trust and play with"),
    ("family", "your mother, father, brothers and sisters together at home"),
    ("home", "the place where you live with your family"),
    ("heart", "it beats in your chest and pumps blood, a symbol of love"),
    ("brain", "the organ in your head that you think with"),
    ("ear", "you hear sounds with it"),
    ("mouth", "you eat and speak with it"),
    ("hand", "it has five fingers and you hold things with it"),
    ("face", "your eyes, nose and mouth are on it"),
    ("joy", "a feeling of great happiness"),
    ("anger", "a strong feeling when you are mad"),
    ("fear", "the feeling you get when you are scared of danger"),
    ("love", "a warm deep feeling you have for family and friends"),
    ("hate", "a strong feeling of dislike, the opposite of love"),
    ("machine", "a thing with moving parts that does work, like an engine"),
    ("tool", "something you hold to fix or build things, like a hammer"),
    ("math", "numbers, adding and counting at school"),
    ("paper", "thin white sheets you write on"),
    ("picture", "a drawing or photo you look at"),
    ("student", "someone who goes to school to learn"),
    ("voice", "the sound you make when you talk or sing"),
    ("secret", "something hidden that you do not tell anyone"),
    ("silence", "no sound at all, completely quiet"),
    ("stranger", "a person you do not know"),
    ("death", "the end of life"),
    ("birth", "when a baby is born, the start of life"),
    ("crime", "breaking the law, like stealing"),
    ("punishment", "what you get for doing something wrong"),
    ("chain", "metal rings linked together"),
    ("hole", "an empty opening in the ground that you dig"),
    ("team", "a group of players who work together"),
    ("world", "the earth and all the people and places on it"),
    ("kindness", "being nice, gentle and helpful to others"),
    ("sadness", "the feeling when you cry and feel unhappy"),
    ("lesson", "what a teacher teaches in a class"),
]

FILLER = set("""a an the it its is are am be you your yours i me my we our they them
their he she his her who what when where which that this these those to of in on
at by for with and or but so do does not no all very can has have had there how
like get go goes""".split())


def tokens(text):
    return re.findall(r"[a-z]+", text.lower())


def prepare(base, questions):
    qs, kept, dropped = [], 0, []
    for ans, text in questions:
        if ans not in base.G:
            continue
        clues = []
        for w in tokens(text):
            if w in FILLER or w == ans or w in clues:
                continue
            if w in base.idx:
                clues.append(w); kept += 1
            else:
                dropped.append(w)
        if clues:
            qs.append((ans, clues, [0] * len(clues)))
    return qs, kept, dropped


def lookup_ranks(base, qs):
    vals = [set(base.G[w].values()) for w in base.words]
    out = []
    for t, cl, _ in qs:
        s = np.array([len(set(cl) & v) for v in vals], float)
        for c in cl:
            if c in base.G:
                s[base.idx[c]] = -1
        out.append(R.rank_of(s, base.idx[t]))
    return np.array(out)


def answers(F, base, qs):
    uniq = sorted({c for _, cl, _ in qs for c in cl}); col = {c: i for i, c in enumerate(uniq)}
    xs = F.streams(uniq); S = [None] + [F.accumulate(xs, d) for d in range(1, F.D + 1)]
    out = []
    for t, cl, _ in qs:
        cols = [col[c] for c in cl]; prev, d = None, F.D
        for dd in range(1, F.D + 1):
            w = int(np.argmax(F.score(S[dd], cols, cl)))
            if w == prev:
                d = dd; break
            prev = w
        sc = F.score(S[d], cols, cl)
        top = [base.words[i] for i in np.argsort(-sc)[:3]]
        out.append((top, R.rank_of(sc, base.idx[t])))
    return out


def summary(name, ranks):
    print(f"  {name:44} top-1 {np.mean(ranks < .5):6.1%}   top-5 {np.mean(ranks < 5):6.1%}   "
          f"median rank {np.median(ranks) + 1:5.0f}")


if __name__ == "__main__":
    base = T.Brain()
    qs, kept, dropped = prepare(base, QUESTIONS)
    print(f"{len(qs)} human questions; question words the brain knows: {kept} "
          f"({kept / (kept + len(dropped)):.0%}); unknown, dropped: {len(dropped)} "
          f"e.g. {', '.join(sorted(set(dropped))[:12])}\n")
    F = FL.Flow(base)
    before = answers(F, base, qs)
    rng = np.random.default_rng(0)
    order = rng.permutation(base.words)
    qtr = FL.mixed_riddles(base, list(order[:int(0.7 * len(order))]), seed=1)
    F.train(qtr)
    after = answers(F, base, qs)

    summary("lookup (question words in the answer's glyph)", lookup_ranks(base, qs))
    summary("brain, untrained", np.array([r for _, r in before]))
    summary("brain+, pathways learned on synthetic riddles", np.array([r for _, r in after]))
    print()
    texts = {a: t for a, t in QUESTIONS}
    for (ans, cl, _), (top, r) in zip(qs, after):
        mark = "HIT " if r < .5 else f"#{r + 1:<3.0f}"
        print(f"  {mark} {texts[ans]:62} -> {', '.join(top):34} (answer: {ans})")
