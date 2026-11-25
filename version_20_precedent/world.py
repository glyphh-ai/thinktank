"""world.py — invented domains with hidden, drifting policies, and precedents.

A policy is a rule: decision X if the case matches one attribute value (or two),
else Y. The policy can change on a given day. Clerks err at a set rate and the
outcome field records it, so a "failed" precedent means the opposite was right.
"""

import itertools

POOL = """red blue green black white gold silver brown pink grey north south east west
river lake ocean hill stone sand snow rain wind storm cloud fire ice wood steel glass
paper cotton wool silk leather apple lemon orange grape melon bread rice corn salt sugar
honey butter cheese fish chicken horse tiger eagle shark wolf bear mouse rabbit snake frog
table chair door window floor wall roof bridge tower castle church school market garden
farm mine port train truck plane ship bike wheel engine motor rocket lamp clock phone
radio camera piano guitar drum violin hammer knife spoon plate bottle basket box bag
shirt coat hat shoe belt ring crown sword shield arrow flag map book letter pencil card
coin bank shop hotel office doctor nurse pilot farmer driver teacher judge king queen
monday winter summer spring autumn morning evening night moon star planet earth""".split()
TRAIN_WORDS, TEST_WORDS = POOL[0::2], POOL[1::2]


def words(rng, split, n):
    """n distinct single-token English words. Train and test never share one, and
    every example draws fresh ones, so a label can only come from reading the page."""
    return rng.sample(TRAIN_WORDS if split == "train" else TEST_WORDS, n)


class Rule:
    """X when every (attr, value) in `conds` matches, else Y."""

    def __init__(self, conds, x, y):
        self.conds, self.x, self.y = tuple(conds), x, y

    def __call__(self, case):
        return self.x if all(case[a] == v for a, v in self.conds) else self.y

    def attrs(self):
        return [a for a, _ in self.conds]


class World:
    def __init__(self, rng, split, n_attr=4, n_val=3):
        w = words(rng, split, n_attr * (n_val + 1) + 3)
        self.attrs = {w[i * (n_val + 1)]: sorted(w[i * (n_val + 1) + 1:(i + 1) * (n_val + 1)])
                      for i in range(n_attr)}
        self.labels = (w[-3], w[-2])
        self.name = w[-1]
        self._hyps = None

    def case(self, rng):
        return {a: rng.choice(v) for a, v in self.attrs.items()}

    def hypotheses(self):
        """Every one-condition and two-condition rule, both polarities."""
        if self._hyps is None:
            x, y = self.labels
            singles = [((a, v),) for a, vs in self.attrs.items() for v in vs]
            pairs = [((a, v), (b, w))
                     for (a, vs), (b, ws) in itertools.combinations(self.attrs.items(), 2)
                     for v in vs for w in ws]
            self._hyps = [Rule(c, p, q) for c in singles + pairs for p, q in ((x, y), (y, x))]
        return self._hyps

    def random_rule(self, rng, two=False):
        names = rng.sample(list(self.attrs), 2 if two else 1)
        x, y = rng.sample(self.labels, 2)
        return Rule([(a, rng.choice(self.attrs[a])) for a in names], x, y)


class Precedent:
    __slots__ = ("id", "day", "case", "decision", "held", "overruled")

    def __init__(self, day, case, decision, held):
        self.id, self.day, self.case, self.decision, self.held = 0, day, case, decision, held
        self.overruled = False

    def effective(self, labels):
        """What was right, given the outcome."""
        return self.decision if self.held else (labels[1] if self.decision == labels[0] else labels[0])


def overlap(a, b):
    return sum(a[k] == b[k] for k in a)


def retrieve(log, case, k):
    """Live precedents first, then most similar, ties to the newer. Oldest first on the page."""
    ranked = sorted(log, key=lambda p: (not p.overruled, overlap(p.case, case), p.day), reverse=True)
    got = sorted(ranked[:k], key=lambda p: p.day)
    for i, p in enumerate(got, 1):
        p.id = i
    return got


def fmt_case(c):
    return " ".join(f"{a}={v}" for a, v in c.items())


def render(world, precedents, case, day):
    lines = [f"Domain {world.name}. Decisions: {world.labels[0]} or {world.labels[1]}.", "Precedents:"]
    for p in precedents:
        lines.append(f"#{p.id} day {p.day}: {fmt_case(p.case)} -> {p.decision}, "
                     f"{'held' if p.held else 'failed'}")
    lines.append(f"Case day {day}: {fmt_case(case)}")
    return "\n".join(lines) + "\nRuling:"


def observer(world, precedents, case):
    """The ideal observer. Fewest policy changes that explain the precedents;
    returns every decision such an explanation gives for `case`."""
    hyps = world.hypotheses()
    seq = [(p.case, p.effective(world.labels)) for p in precedents]

    def fits(h, part):
        return all(h(c) == d for c, d in part)

    whole = [h for h in hyps if fits(h, seq)]
    if whole:
        return {h(case) for h in whole}
    out = set()
    for t in range(1, len(seq)):
        if any(fits(h, seq[:t]) for h in hyps):
            out |= {h(case) for h in hyps if fits(h, seq[t:])}
    return out


def nearest(precedents, case):
    return max(precedents, key=lambda p: (overlap(p.case, case), p.day))


def gold_move(world, precedents, case, now_rule):
    """The right move, given the true current policy."""
    want = now_rule(case)
    n = nearest(precedents, case)
    labels = world.labels
    if n.effective(labels) == want:
        if n.held:
            return "apply", f"apply #{n.id}"
        return "reverse", f"overrule #{n.id} by outcome"
    if now_rule(n.case) != want:
        diff = [a for a in now_rule.attrs() if n.case[a] != case[a]]
        return "distinguish", f"distinguish #{n.id} on {diff[0]}"
    same = [p for p in precedents if now_rule(p.case) == want and p.effective(labels) == want
            and p.day > n.day]
    if not same:
        return None, None
    return "overrule", f"overrule #{n.id} by #{same[-1].id}"


def sample(rng, split, want_kind=None, n_attr=4, k=10):
    """One identifiable example, or None. Rejection sampling; the caller retries."""
    w = World(rng, split, n_attr=n_attr)
    two = rng.random() < 0.3
    old = w.random_rule(rng, two)
    change = rng.random() < 0.7
    new = w.random_rule(rng, two) if change else old
    days = sorted(rng.sample(range(1, 200), 40))
    cut = rng.choice(days[12:30])
    log = []
    for d in days:
        c = w.case(rng)
        right = (old if d < cut else new)(c)
        err = rng.random() < 0.12
        wrong = w.labels[1] if right == w.labels[0] else w.labels[0]
        log.append(Precedent(d, c, wrong if err else right, not err))
    case = w.case(rng)
    got = retrieve(log, case, k)
    kind, move = gold_move(w, got, case, new)
    if kind is None or (want_kind and kind != want_kind):
        return None
    if observer(w, got, case) != {new(case)}:
        return None
    return dict(prompt=render(w, got, case, 201), decision=new(case), kind=kind, move=move,
                got=[(p.id, p.day, p.case, p.decision, p.held) for p in got], case=case,
                labels=w.labels)
