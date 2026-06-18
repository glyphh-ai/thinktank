"""
big_loop.py — the in-loop test at scale, on Ada's REAL enriched data.

Uses glyphh-ai/adaL's actual enricher output (benchmark/curriculum/agi/ada.jsonl:
~39k facts across chemistry / philosophy / law, mapped into the universal schema).
We render each fact to NL and tag every token with the Ada slot it came from,
then run the same in-loop test as prove.py but on real cross-domain vocabulary
and real role ambiguity, at scale.

No API key needed — these are pre-computed enricher outputs in the repo.

Run:  python big_loop.py
"""

import os, json, subprocess
import numpy as np
import ada_loop as A
import prove as P

ADAL = "/tmp/adaL"
DATA = f"{ADAL}/benchmark/curriculum/agi/ada.jsonl"
MAX_TOK_PER_VALUE = 4
N_FACTS = 14000
VOCAB_CAP = 8000


def ensure_adaL():
    if not os.path.exists(DATA):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/glyphh-ai/adaL", ADAL],
                       check=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})


def toks(val):
    return [t for t in str(val).lower().replace("_", "-").replace("-", " ").split()
            if t][:MAX_TOK_PER_VALUE]


def render(rec):
    """Fact -> list of (word, ada_role). None if not renderable."""
    e = rec.get("entity") or {}
    r = rec.get("relational") or {}
    name = e.get("name")
    if not name:
        return None
    out = [(w, "entity.name") for w in toks(name)]
    if r.get("predicate") and r.get("object"):
        out += [(w, "relational.predicate") for w in toks(r["predicate"])]
        out += [(w, "relational.object") for w in toks(r["object"])]
    elif e.get("kind"):
        out += [("is", "O"), ("a", "O")] + [(w, "entity.kind") for w in toks(e["kind"])]
    else:
        return None
    # a couple more slots when present -> richer role variety
    if e.get("subkind"):
        out += [("the", "O")] + [(w, "entity.subkind") for w in toks(e["subkind"])]
    return out if len(out) >= 3 else None


def build_corpus():
    ensure_adaL()
    facts = []
    with open(DATA) as f:
        for line in f:
            if len(facts) >= N_FACTS:
                break
            try:
                r = render(json.loads(line))
            except Exception:
                continue
            if r:
                facts.append(r)
    rng = np.random.default_rng(0); rng.shuffle(facts)

    # vocab cap: keep the top words, rest -> <unk>
    from collections import Counter
    wc = Counter(w for f in facts for w, _ in f)
    keep = {w for w, _ in wc.most_common(VOCAB_CAP)}
    toks_, roles_ = [], []
    for f in facts:
        for w, role in f:
            toks_.append(w if w in keep else "<unk>"); roles_.append(role)
        toks_.append("<sep>"); roles_.append("O")
    vocab = sorted(set(toks_)); stoi = {w: i for i, w in enumerate(vocab)}
    ids = np.array([stoi[w] for w in toks_], dtype=np.int64)
    rids = np.array([A.ROLE_ID[r] for r in roles_], dtype=np.int64)
    return ids, rids, vocab, stoi, len(facts)


def lm_acc(p, va, rng, B=256, reps=6):
    accs = []
    for _ in range(reps):
        x, yw, yr = P.batch_tag(va, rng, B=B)
        lm, _, _ = A.forward(p, x)
        accs.append((lm.argmax(-1) == yw).mean())
    return np.mean(accs)


if __name__ == "__main__":
    ids, rids, vocab, stoi, nf = build_corpus()
    n = int(0.9*len(ids)); tr=(ids[:n], rids[:n]); va=(ids[n:], rids[n:])
    ambig = P.ambiguous_words(ids, rids, vocab)
    lut = P.lookup_table(*tr)
    roles_used = sorted(set(A.ROLES[r] for r in rids) - {"O"})
    print(f"REAL Ada data: {nf} facts -> {len(ids)} tokens, vocab={len(vocab)} "
          f"(capped {VOCAB_CAP}), ambiguous words={len(ambig)}")
    print(f"Ada roles in play: {roles_used}\n")

    print("training in-loop on real cross-domain data (2 seeds, 1500 steps)...")
    m_all, m_amb, m_un, lm1, lm0 = [], [], [], [], []
    for sd in (0, 1):
        p = P.train_tag(tr, vocab, steps=1500, lam=1.0, seed=sd)
        mok, lok, tot, acc = P.detailed_eval(p, va, lut, ambig, np.random.default_rng(99))
        m_all.append((mok[True]+mok[False])/(tot[True]+tot[False]))
        m_amb.append(acc(mok, True)); m_un.append(acc(mok, False))
        lm1.append(lm_acc(p, va, np.random.default_rng(7)))
    # LM-only baseline (lambda=0) to check the role objective doesn't hurt LM
    p0 = P.train_tag(tr, vocab, steps=1500, lam=0.0, seed=0)
    lm0.append(lm_acc(p0, va, np.random.default_rng(7)))
    look_all = (lok[True]+lok[False])/(tot[True]+tot[False])
    look_amb = acc(lok, True); look_un = acc(lok, False)

    print(f"\n{'role accuracy (fair tagging)':30} {'overall':>9} {'AMBIGUOUS':>11} {'unambiguous':>12}")
    print(f"{'word->role lookup (baseline)':30} {look_all:9.1%} {look_amb:11.1%} {look_un:12.1%}")
    print(f"{'Ada-in-the-loop model':30} {np.mean(m_all):9.1%} {np.mean(m_amb):11.1%} {np.mean(m_un):12.1%}")
    print(f"\nambiguous-token gap: model {np.mean(m_amb):.1%} vs lookup {look_amb:.1%} "
          f"(+{np.mean(m_amb)-look_amb:.1%})")
    print(f"LM next-word acc: lambda=1 {np.mean(lm1):.1%}  vs  lambda=0 {lm0[0]:.1%}  "
          f"(role objective {'free' if abs(np.mean(lm1)-lm0[0])<0.02 else 'has a cost'})")
