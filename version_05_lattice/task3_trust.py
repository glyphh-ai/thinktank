"""
task3_trust.py — does Ada-in-the-loop buy TRUST (knowing what it doesn't know)?

Proving the model can EMIT Ada's slots (prove.py, big_loop.py) isn't enough; the
real question is whether the structure makes the model more trustworthy. Ada's
signature is refusing on an empty slot instead of confabulating. So we test
calibration:

  - train on KNOWN (entity, slot) facts only; hold out some pairs the model was
    NEVER told (knows carol's city, never told carol's job).
  - query answerable (seen) vs unanswerable (held-out) pairs; read confidence
    at the answer position.
  - metric: AUROC of confidence separating answerable from unanswerable -- does
    the model get LESS confident when it genuinely doesn't know?

Clean ablation: identical corpus + model; the ONLY difference is the Ada role
objective (lambda=1 vs lambda=0). Higher AUROC at equal answerable-accuracy =>
the structure buys calibrated trust.

Run:  python task3_trust.py
"""

import numpy as np
import ada_loop as A
import prove as P

NAMES = [f"e{i}" for i in range(80)]
SLOTS = {  # slot -> (prompt words, value pool, Ada role of the value)
    "job":    (["works","as","a"],          ["engineer","teacher","doctor","chef","pilot","lawyer","nurse","artist"], "relational.object"),
    "city":   (["lives","in"],              ["austin","denver","boston","miami","seattle","chicago","dallas","phoenix"], "spatial.location"),
    "origin": (["is","from"],               ["ohio","texas","maine","cuba","oregon","iowa","idaho","kansas"], "spatial.origin"),
    "pet":    (["owns","a"],                ["parrot","dog","cat","fish","rabbit","snake","horse","mouse"], "relational.object"),
    "color":  (["likes","the","color"],     ["green","brown","black","gold","white","red","blue","grey"], "perceptual.color"),
}


def build(seed=0, repeat=60, known_per=3):
    rng = np.random.default_rng(seed)
    known, unknown = [], []          # (name, slot, value)
    facts = []
    for nm in NAMES:
        order = list(SLOTS); rng.shuffle(order)
        for j, sl in enumerate(order):
            pool = SLOTS[sl][1]; val = pool[rng.integers(len(pool))]
            if j < known_per:
                known.append((nm, sl, val))
            else:
                unknown.append((nm, sl, val))   # held out: never trained
    # training stream: only KNOWN facts (with role tags), repeated/shuffled
    toks, roles = [], []
    for _ in range(repeat):
        rng.shuffle(known)
        for nm, sl, val in known:
            pw, _, vr = SLOTS[sl]
            seq = [(nm, "entity.name")] + [(w, "O") for w in pw] + [(val, vr)]
            for w, r in seq: toks.append(w); roles.append(r)
            toks.append("<sep>"); roles.append("O")
    vocab = sorted(set(toks)); stoi = {w: i for i, w in enumerate(vocab)}
    ids = np.array([stoi[w] for w in toks]); rids = np.array([A.ROLE_ID[r] for r in roles])
    return ids, rids, vocab, stoi, known, unknown


def prompt_ids(name, slot, stoi):
    pw = SLOTS[slot][0]
    seq = [name] + pw
    return [stoi[w] for w in seq if w in stoi]


def confidence_and_answer(p, name, slot, val, stoi):
    ids = prompt_ids(name, slot, stoi)
    if not ids: return None
    lm, _, _ = A.forward(p, np.array([ids]))
    z = lm[0, -1]; pr = np.exp(z - z.max()); pr /= pr.sum()
    conf = float(pr.max())
    correct = (val in stoi) and (int(pr.argmax()) == stoi[val])
    return conf, correct


def auroc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    allv = np.concatenate([pos, neg]); order = allv.argsort()
    ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(allv)+1)
    rp = ranks[:len(pos)].sum()
    return (rp - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))


def evaluate(p, known, unknown, stoi):
    ck = [confidence_and_answer(p, *kk[:2], kk[2], stoi) for kk in known]
    cu = [confidence_and_answer(p, *uu[:2], uu[2], stoi) for uu in unknown]
    ck = [c for c in ck if c]; cu = [c for c in cu if c]
    known_acc = np.mean([c[1] for c in ck])
    conf_known = [c[0] for c in ck]; conf_unknown = [c[0] for c in cu]
    return known_acc, np.mean(conf_known), np.mean(conf_unknown), auroc(conf_known, conf_unknown)


if __name__ == "__main__":
    ids, rids, vocab, stoi, known, unknown = build()
    tr = (ids, rids)
    print(f"entities={len(NAMES)} slots={len(SLOTS)} | known facts={len(known)} "
          f"(trained), held-out unknown={len(unknown)} | vocab={len(vocab)}")
    print("test: is the model LESS confident on (entity,slot) it was never told?\n")

    print(f"{'config':22} {'known acc':>10} {'conf(known)':>12} {'conf(unknown)':>14} {'AUROC':>8}")
    res = {}
    for name, lam in (("plain LM (lambda=0)", 0.0), ("Ada-in-loop (lambda=1)", 1.0)):
        ka, ckn, cuk, auc = [], [], [], []
        for sd in (0, 1, 2):
            p = P.train_tag(tr, vocab, steps=1500, lam=lam, seed=sd)
            a, k, u, ar = evaluate(p, known, unknown, stoi)
            ka.append(a); ckn.append(k); cuk.append(u); auc.append(ar)
        res[name] = np.mean(auc)
        print(f"{name:22} {np.mean(ka):10.1%} {np.mean(ckn):12.2f} {np.mean(cuk):14.2f} {np.mean(auc):8.3f}")

    d = res["Ada-in-loop (lambda=1)"] - res["plain LM (lambda=0)"]
    print(f"\nAUROC delta (Ada-in-loop - plain): {d:+.3f}")
    print("AUROC 0.5 = can't tell known from unknown; >0.5 = less confident when it"
          " doesn't know.\n=> structure buys trust IFF Ada-in-loop AUROC is clearly higher.")
