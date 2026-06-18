"""
prove.py — does the model really learn Ada's STRUCTURE, or just memorize word->role?

The killer test (Ada's own benchmark philosophy): build genuine role ambiguity,
where the SAME word fills different Ada slots depending on context:
  - a city is spatial.location after "lives in", spatial.origin after "is from"
  - a pet  is relational.object after "owns a", entity.name in "the <pet> is ..."

Then compare the in-loop model against a word->majority-role LOOKUP baseline.
A lookup table is exactly what wins if the model only memorized. If the model
beats lookup ON AMBIGUOUS TOKENS, it must be using context -> it learned the
structure, not the surface word.

Run:  python prove.py
"""

import numpy as np
import ada_loop as A

NAMES  = ["carol","dave","erin","frank","gina","hank","ivy","jack","kate","leo",
          "mona","ned","olga","pete","quinn","rosa","sam","tina","uma","viktor"]
CITIES = ["austin","denver","boston","miami","seattle","chicago","portland","dallas"]
PETS   = ["parrot","dog","cat","fish","rabbit","snake"]
COLORS = ["green","brown","black","gold","white","red","blue","grey"]


def build(n_each=240, seed=0):
    """Generate facts so cities appear ~equally as location/origin and pets
    ~equally as object/name -> those words are genuinely ambiguous."""
    rng = np.random.default_rng(seed)
    facts = []
    pick = lambda xs: xs[rng.integers(len(xs))]
    for _ in range(n_each):
        facts.append([(pick(NAMES),"entity.name"),("lives","relational.predicate"),("in","O"),(pick(CITIES),"spatial.location")])
        facts.append([(pick(NAMES),"entity.name"),("is","O"),("from","O"),(pick(CITIES),"spatial.origin")])
        facts.append([(pick(NAMES),"entity.name"),("owns","relational.predicate"),("a","O"),(pick(PETS),"relational.object")])
        facts.append([("the","O"),(pick(PETS),"entity.name"),("is","O"),(pick(COLORS),"perceptual.color")])
    rng.shuffle(facts)
    toks, roles = [], []
    for f in facts:
        for w, r in f: toks.append(w); roles.append(r)
        toks.append("<sep>"); roles.append("O")
    vocab = sorted(set(toks)); stoi = {w:i for i,w in enumerate(vocab)}
    ids  = np.array([stoi[w] for w in toks], dtype=np.int64)
    rids = np.array([A.ROLE_ID[r] for r in roles], dtype=np.int64)
    return ids, rids, vocab, stoi


def ambiguous_words(ids, rids, vocab):
    """Words that appear with >1 distinct Ada role in the corpus."""
    seen = {}
    for w, r in zip(ids, rids):
        seen.setdefault(w, set()).add(r)
    return {w for w, rs in seen.items() if len(rs) > 1}


def lookup_table(ids, rids):
    """word -> its single most-common role id (the memorization baseline)."""
    from collections import Counter, defaultdict
    c = defaultdict(Counter)
    for w, r in zip(ids, rids): c[w][r] += 1
    return {w: cc.most_common(1)[0][0] for w, cc in c.items()}


def batch_tag(data, rng, B=32):
    """FAIR tagging setup: role target is the CURRENT token's role (both the
    model and the lookup see the same token). LM target stays the next token."""
    ids, rids = data
    ix = rng.integers(0, len(ids) - A.T - 1, size=B)
    x  = np.stack([ids[i:i+A.T]      for i in ix])
    yw = np.stack([ids[i+1:i+1+A.T]  for i in ix])   # LM: next word
    yr = np.stack([rids[i:i+A.T]     for i in ix])   # ROLE: tag CURRENT token
    return x, yw, yr


def train_tag(tr, vocab, steps=1200, lam=1.0, seed=0, B=32, lr=3e-3):
    rng = np.random.default_rng(seed)
    p = A.init_params(rng, len(vocab))
    opt = A.Adam(p, lr=lr)
    brng = np.random.default_rng(seed + 7)
    for _ in range(steps):
        x, yw, yr = batch_tag(tr, brng, B=B)
        _, _, _, g = A.loss_and_grad(p, x, yw, yr, lam=lam)
        opt.step(p, g)
    return p


def detailed_eval(p, va, lut, ambig, rng, B=256, reps=8):
    """Role accuracy of MODEL vs LOOKUP, split by ambiguous / unambiguous word.
    Both condition on the SAME current token -> fair; only the model has context."""
    mok = {True: 0, False: 0}; lok = {True: 0, False: 0}; tot = {True: 0, False: 0}
    for _ in range(reps):
        x, yw, yr = batch_tag(va, rng, B=B)
        _, role, _ = A.forward(p, x)
        mpred = role.argmax(-1)
        for b in range(x.shape[0]):
            for t in range(x.shape[1]):
                w = int(x[b, t]); true = int(yr[b, t])      # current token + its role
                amb = w in ambig
                tot[amb] += 1
                mok[amb] += int(mpred[b, t] == true)
                lok[amb] += int(lut.get(w, A.ROLE_ID["O"]) == true)
    def acc(d, amb): return d[amb] / max(1, tot[amb])
    return mok, lok, tot, acc


if __name__ == "__main__":
    ids, rids, vocab, stoi = build()
    n = int(0.9*len(ids)); tr=(ids[:n], rids[:n]); va=(ids[n:], rids[n:])
    ambig = ambiguous_words(ids, rids, vocab)
    lut = lookup_table(*tr)
    amb_words = sorted(vocab[w] for w in ambig)
    print(f"corpus: {len(ids)} tokens, vocab={len(vocab)}")
    print(f"ambiguous words ({len(amb_words)}): {amb_words}\n")

    print("training Ada-in-the-loop, fair tagging (3 seeds)...")
    macc_all, macc_amb, macc_un = [], [], []
    for sd in (0, 1, 2):
        p = train_tag(tr, vocab, steps=1200, lam=1.0, seed=sd)
        mok, lok, tot, acc = detailed_eval(p, va, lut, ambig, np.random.default_rng(99))
        macc_all.append((mok[True]+mok[False])/(tot[True]+tot[False]))
        macc_amb.append(acc(mok, True))
        macc_un.append(acc(mok, False))
    # lookup is deterministic; compute once from the last run's buckets
    look_all = (lok[True]+lok[False])/(tot[True]+tot[False])
    look_amb = acc(lok, True)
    look_un  = acc(lok, False)

    print(f"\n{'role accuracy (fair tagging)':28} {'overall':>9} {'AMBIGUOUS':>11} {'unambiguous':>12}")
    print(f"{'word->role lookup (baseline)':28} {look_all:9.1%} {look_amb:11.1%} {look_un:12.1%}")
    print(f"{'Ada-in-the-loop model':28} {np.mean(macc_all):9.1%} {np.mean(macc_amb):11.1%} {np.mean(macc_un):12.1%}")
    print(f"\n=> on AMBIGUOUS tokens: model {np.mean(macc_amb):.1%} vs lookup {look_amb:.1%} "
          f"(+{np.mean(macc_amb)-look_amb:.1%})")
    print("   the model can ONLY win here by using context -> it learned Ada's structure.")
