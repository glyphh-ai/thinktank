"""
ada_loop.py — Ada's schema IN the training loop (not a post-hoc read-out).
==========================================================================
Word-level GPT over a corpus of short facts. Two heads share the trunk:

  1. LM head   -> predict the next word            (cross-entropy)
  2. ROLE head -> predict the next word's Ada slot  (cross-entropy)
                  i.e. which of Ada's 7-layer x 33-role lattice slots the
                  next word fills (entity.name / spatial.location / ...)

  total_loss = LM_CE  +  lambda * ROLE_CE

Putting the Ada slot in the LOSS makes the structured meaning load-bearing in
the gradient -- it can't wash out (v04 showed an input channel just gets
ignored), and the "why" is GENERATED in the loop: the model emits each token's
Ada role itself, every step.

Ada schema vendored from glyphh-ai/adaL (Apache-2.0).

Run:  python ada_loop.py
"""

import numpy as np
from step3_gpt import (layernorm_fwd, layernorm_bwd, linear_fwd, linear_bwd,
                       attention_fwd, attention_bwd, Adam, C)

T = 12          # context window (words); facts are short
np.seterr(over="ignore")

# ── Ada universal schema (vendored from glyphh-ai/adaL, Apache-2.0) ──────────
UNIVERSAL_SCHEMA = {
    "entity":       ["name", "kind", "subkind"],
    "perceptual":   ["color", "size", "shape", "texture", "sound",
                     "smell", "taste", "temperature"],
    "spatial":      ["location", "origin", "direction"],
    "temporal":     ["time", "duration", "age", "era", "frequency"],
    "relational":   ["subject", "predicate", "object", "possessor",
                     "agent", "patient", "instrument"],
    "quantitative": ["count", "magnitude", "unit", "ratio"],
    "epistemic":    ["source", "certainty", "modality"],
}
ALL_ROLES = [f"{L}.{r}" for L, rs in UNIVERSAL_SCHEMA.items() for r in rs]
ROLES = ["O"] + ALL_ROLES                      # "O" = filler/function word
ROLE_ID = {r: i for i, r in enumerate(ROLES)}
N_ROLE = len(ROLES)                            # 34

# ── Fact corpus: each content word is tagged with the Ada role it fills ──────
ENTITIES = [
    dict(name="carol", loc="austin",   origin="ohio",     job="engineer", pet="parrot", color="green", age="thirty", count="two"),
    dict(name="dave",  loc="denver",   origin="texas",    job="teacher",  pet="dog",    color="brown", age="forty",  count="one"),
    dict(name="erin",  loc="boston",   origin="maine",    job="doctor",   pet="cat",    color="black", age="fifty",  count="three"),
    dict(name="frank", loc="miami",    origin="cuba",     job="chef",     pet="fish",   color="gold",  age="sixty",  count="four"),
    dict(name="gina",  loc="seattle",  origin="oregon",   job="pilot",    pet="rabbit", color="white", age="twenty", count="five"),
    dict(name="hank",  loc="chicago",  origin="iowa",     job="lawyer",   pet="snake",  color="red",   age="thirty", count="one"),
    dict(name="ivy",   loc="portland", origin="idaho",    job="nurse",    pet="bird",   color="blue",  age="forty",  count="two"),
    dict(name="jack",  loc="dallas",   origin="kansas",   job="farmer",   pet="horse",  color="grey",  age="fifty",  count="six"),
    dict(name="kate",  loc="phoenix",  origin="arizona",  job="artist",   pet="turtle", color="green", age="twenty", count="one"),
    dict(name="leo",   loc="atlanta",  origin="georgia",  job="banker",   pet="mouse",  color="brown", age="sixty",  count="two"),
]

def _fact(e):
    """Return a list of (word, role) for a random template filled by entity e."""
    n = ("name", "entity.name")
    templates = [
        lambda: [(e["name"],"entity.name"), ("lives","relational.predicate"), ("in","O"), (e["loc"],"spatial.location")],
        lambda: [(e["name"],"entity.name"), ("is","O"), ("from","O"), (e["origin"],"spatial.origin")],
        lambda: [(e["name"],"entity.name"), ("works","relational.predicate"), ("as","O"), ("a","O"), (e["job"],"relational.object")],
        lambda: [(e["name"],"entity.name"), ("owns","relational.predicate"), ("a","O"), (e["pet"],"relational.object")],
        lambda: [("the","O"), (e["pet"],"entity.name"), ("is","O"), (e["color"],"perceptual.color")],
        lambda: [(e["name"],"entity.name"), ("is","O"), (e["age"],"temporal.age"), ("years","quantitative.unit"), ("old","O")],
        lambda: [(e["name"],"entity.name"), ("likes","relational.predicate"), ("the","O"), ("color","O"), (e["color"],"perceptual.color")],
        lambda: [(e["name"],"entity.name"), ("has","relational.predicate"), (e["count"],"quantitative.count"), ("cats","relational.object")],
    ]
    return templates

def build_corpus(repeat=40, seed=0):
    rng = np.random.default_rng(seed)
    toks, roles = [], []
    facts = [(e, t) for e in ENTITIES for t in range(len(_fact(ENTITIES[0])))]
    for _ in range(repeat):
        rng.shuffle(facts)
        for e, ti in facts:
            for w, r in _fact(e)[ti]():
                toks.append(w); roles.append(r)
            toks.append("<sep>"); roles.append("O")
    vocab = sorted(set(toks))
    stoi = {w: i for i, w in enumerate(vocab)}
    ids = np.array([stoi[w] for w in toks], dtype=np.int64)
    rids = np.array([ROLE_ID[r] for r in roles], dtype=np.int64)
    return ids, rids, vocab, stoi


# ── Model: trunk (reused blocks) + two heads ────────────────────────────────
def init_params(rng, vocab, C=C):
    def n(*s, sc): return rng.normal(0, sc, size=s)
    return {
        "tok_emb": n(vocab, C, sc=0.02), "pos_emb": n(T, C, sc=0.02),
        "Wq": n(C, C, sc=1/np.sqrt(C)), "Wk": n(C, C, sc=1/np.sqrt(C)),
        "Wv": n(C, C, sc=1/np.sqrt(C)), "Wo": n(C, C, sc=1/np.sqrt(C)),
        "g1": np.ones(C), "b1": np.zeros(C), "g2": np.ones(C), "b2": np.zeros(C),
        "gf": np.ones(C), "bf": np.zeros(C),
        "W1": n(C, 4*C, sc=1/np.sqrt(C)), "b1m": np.zeros(4*C),
        "W2": n(4*C, C, sc=1/np.sqrt(4*C)), "b2m": np.zeros(C),
        "W_lm": n(C, vocab, sc=1/np.sqrt(C)), "b_lm": np.zeros(vocab),
        "W_role": n(C, N_ROLE, sc=1/np.sqrt(C)), "b_role": np.zeros(N_ROLE),
    }

def forward(p, x):
    B, Tt = x.shape
    h0 = p["tok_emb"][x] + p["pos_emb"][:Tt]
    ln1, c1 = layernorm_fwd(h0, p["g1"], p["b1"])
    attn, ca = attention_fwd(ln1, p); h1 = h0 + attn
    ln2, c2 = layernorm_fwd(h1, p["g2"], p["b2"])
    hid, cl1 = linear_fwd(ln2, p["W1"], p["b1m"]); relu = hid > 0; act = hid*relu
    mlp, cl2 = linear_fwd(act, p["W2"], p["b2m"]); h2 = h1 + mlp
    lnf, cf = layernorm_fwd(h2, p["gf"], p["bf"])
    lm, chlm = linear_fwd(lnf, p["W_lm"], p["b_lm"])
    role, chr_ = linear_fwd(lnf, p["W_role"], p["b_role"])
    cache = dict(x=x, B=B, T=Tt, c1=c1, ca=ca, c2=c2, cl1=cl1, relu=relu,
                 cl2=cl2, cf=cf, chlm=chlm, chr=chr_)
    return lm, role, cache

def _ce(logits, y):
    V = logits.shape[-1]; flat = logits.reshape(-1, V) - logits.reshape(-1, V).max(1, keepdims=True)
    pr = np.exp(flat); pr /= pr.sum(1, keepdims=True)
    yf = y.reshape(-1); N = yf.shape[0]
    loss = -np.log(pr[np.arange(N), yf] + 1e-12).mean()
    d = pr.copy(); d[np.arange(N), yf] -= 1; d /= N
    return loss, d.reshape(logits.shape)

def loss_and_grad(p, x, yw, yr, lam=1.0):
    lm, role, cache = forward(p, x)
    lm_loss, dlm = _ce(lm, yw)
    role_loss, drole = _ce(role, yr); drole *= lam
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    dlnf_a, grads["W_lm"], grads["b_lm"] = linear_bwd(dlm, cache["chlm"])
    dlnf_b, grads["W_role"], grads["b_role"] = linear_bwd(drole, cache["chr"])
    dlnf = dlnf_a + dlnf_b
    dh2, grads["gf"], grads["bf"] = layernorm_bwd(dlnf, cache["cf"])
    dh1 = dh2.copy()
    dact, grads["W2"], grads["b2m"] = linear_bwd(dh2, cache["cl2"])
    dhid = dact * cache["relu"]
    dln2, grads["W1"], grads["b1m"] = linear_bwd(dhid, cache["cl1"])
    dh1b, grads["g2"], grads["b2"] = layernorm_bwd(dln2, cache["c2"]); dh1 += dh1b
    dh0 = dh1.copy()
    dln1, gatt = attention_bwd(dh1, cache["ca"], p)
    for k, v in gatt.items(): grads[k] += v
    dh0b, grads["g1"], grads["b1"] = layernorm_bwd(dln1, cache["c1"]); dh0 += dh0b
    np.add.at(grads["tok_emb"], cache["x"], dh0)
    grads["pos_emb"][:cache["T"]] += dh0.sum(0)
    return (lm_loss + lam*role_loss), lm_loss, role_loss, grads


def get_batch(ids, rids, rng, B=32):
    ix = rng.integers(0, len(ids) - T - 1, size=B)
    x  = np.stack([ids[i:i+T]       for i in ix])
    yw = np.stack([ids[i+1:i+1+T]   for i in ix])
    yr = np.stack([rids[i+1:i+1+T]  for i in ix])
    return x, yw, yr


def evaluate(p, va, rng, B=128, reps=6):
    lm_acc, role_acc = [], []
    for _ in range(reps):
        x, yw, yr = get_batch(*va, rng, B=B)
        lm, role, _ = forward(p, x)
        lm_acc.append((lm.argmax(-1) == yw).mean())
        role_acc.append((role.argmax(-1) == yr).mean())
    return np.mean(lm_acc), np.mean(role_acc)


def train(tr, va, vocab, steps=800, lam=1.0, seed=0, B=32, lr=3e-3):
    rng = np.random.default_rng(seed)
    p = init_params(rng, len(vocab))
    opt = Adam(p, lr=lr)
    brng = np.random.default_rng(seed + 7)
    erng = np.random.default_rng(123)
    for s in range(1, steps + 1):
        x, yw, yr = get_batch(*tr, brng, B=B)
        _, _, _, g = loss_and_grad(p, x, yw, yr, lam=lam)
        opt.step(p, g)
    la, ra = evaluate(p, va, erng)
    return p, la, ra


def why_demo(p, vocab, stoi, contexts):
    itos = {i: w for w, i in stoi.items()}
    print("in-loop 'why' — model emits next word AND its Ada slot, per step:")
    for ctx in contexts:
        ids = [stoi[w] for w in ctx.split() if w in stoi][-T:]
        lm, role, _ = forward(p, np.array([ids]))
        w = itos[int(lm[0, -1].argmax())]
        r = ROLES[int(role[0, -1].argmax())]
        print(f"  {ctx:24} -> {w:10} [{r}]")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ids, rids, vocab, stoi = build_corpus()
    n = int(0.9*len(ids)); tr=(ids[:n], rids[:n]); va=(ids[n:], rids[n:])
    print(f"corpus: {len(ids)} word-tokens, vocab={len(vocab)}, roles={N_ROLE}")
    print(f"Ada roles in play: {sorted(set(ROLES[r] for r in rids) - {'O'})}\n")

    p0 = init_params(rng, len(vocab))
    gx, gw, gr = get_batch(*tr, np.random.default_rng(1), B=4)
    _,_,_, G = loss_and_grad(p0, gx, gw, gr, lam=1.0)
    worst = 0; h=1e-5; cr=np.random.default_rng(2)
    for k in p0:
        f=p0[k].reshape(-1); idx=cr.choice(f.size, size=min(5,f.size), replace=False)
        for i in idx:
            o=f[i]; f[i]=o+h; lp=loss_and_grad(p0,gx,gw,gr)[0]; f[i]=o-h; lmv=loss_and_grad(p0,gx,gw,gr)[0]; f[i]=o
            num=(lp-lmv)/(2*h); worst=max(worst, abs(num-G[k].reshape(-1)[i])/max(1e-9,abs(num)+abs(G[k].reshape(-1)[i])))
    print(f"dual-head gradient check: worst rel err {worst:.2e}  ({'OK' if worst<1e-4 else 'FAIL'})\n")

    print("training (3 seeds each)...")
    print(f"{'config':28} {'LM next-word acc':>17} {'Ada role acc':>14}")
    for name, lam in (("LM only (lambda=0)", 0.0), ("Ada in the loop (lambda=1)", 1.0)):
        las, ras = [], []
        for sd in (0, 1, 2):
            _, la, ra = train(tr, va, vocab, steps=800, lam=lam, seed=sd)
            las.append(la); ras.append(ra)
        print(f"{name:28} {np.mean(las):16.1%} {np.mean(ras):14.1%}")

    print()
    p, _, _ = train(tr, va, vocab, steps=800, lam=1.0, seed=0)
    why_demo(p, vocab, stoi,
             ["carol lives in", "dave is from", "erin works as a",
              "the parrot is", "gina has", "jack is"])

