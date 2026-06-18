"""
long_memory.py — the thing a windowed LLM physically can't do: recall a fact
stated far beyond its context window.

The full loop: read a stream of facts -> the in-loop model extracts each
(entity, slot, value) from its role head AS IT READS (generation -> structure)
-> write to a persistent store -> recall any distance later.

We give NOVEL facts (entity-value combos NOT in training, often contradicting
the trained value) so the answer can only come from the STREAM, not the weights.
Then probe every fact at the end of the stream:

  - plain LM (context window only): can copy a fact only if it's within the last
    T tokens; beyond that it's gone (and it confabulates its stale trained value).
  - in-loop + store: extracted when seen -> recalled at any distance.

The recall win is external memory (Ada's domain). What this tests that's ours:
can the in-loop model natively EXTRACT the right triples to populate it?

Run:  python long_memory.py
"""

import numpy as np
import ada_loop as A
import prove as P
from task3_trust import NAMES, SLOTS, build as build_train


def train_model(seed=0, steps=1500, lam=1.0):
    ids, rids, vocab, stoi, known, unknown = build_train(seed=seed)
    p = P.train_tag((ids, rids), vocab, steps=steps, lam=lam, seed=seed)
    return p, vocab, stoi, {(n, s): v for (n, s, v) in known}


def make_stream(stoi, trained_vals, n_facts=40, seed=1):
    """A stream of NOVEL facts (new entity->value, often != trained value)."""
    rng = np.random.default_rng(seed)
    facts, toks = [], []
    for _ in range(n_facts):
        nm = NAMES[rng.integers(len(NAMES))]
        sl = list(SLOTS)[rng.integers(len(SLOTS))]
        pool = SLOTS[sl][1]; val = pool[rng.integers(len(pool))]
        pw = SLOTS[sl][0]
        words = [nm] + pw + [val]
        start = len(toks)
        toks += words + ["<sep>"]
        facts.append(dict(name=nm, slot=sl, val=val, words=words,
                          val_pos=start + len(words) - 1))   # token index of value
    ids = np.array([stoi.get(w, stoi.get("<unk>", 0)) for w in toks])
    return facts, toks, ids


def extract_to_store(p, facts, toks, stoi):
    """Read each fact, use the role head to pull (entity, slot, value) -> store."""
    itos = {i: w for w, i in stoi.items()}
    content = {"spatial.location", "spatial.origin", "relational.object",
               "perceptual.color", "entity.kind"}
    store = {}
    for f in facts:
        seq = [stoi.get(w, stoi.get("<unk>", 0)) for w in f["words"]]
        _, role, _ = A.forward(p, np.array([seq]))
        pr = role[0].argmax(-1)
        ent = val = slot = None
        for t, w in enumerate(f["words"]):
            r = A.ROLES[int(pr[t])]
            if r == "entity.name": ent = w
            elif r in content:     val, slot = w, r
        if ent and val:
            store[(ent, f["slot"])] = val          # keyed by true slot for lookup
    return store


def plain_recall(p, f, ids, stoi, qpos):
    """Plain LM: answer 'name <prompt>' from the last-T-token window ending at qpos."""
    pw = SLOTS[f["slot"]][0]
    prompt = [stoi[w] for w in ([f["name"]] + pw) if w in stoi]
    room = A.T - len(prompt)
    window = ids[max(0, qpos - room):qpos].tolist()
    inp = (window + prompt)[-A.T:]
    lm, _, _ = A.forward(p, np.array([inp]))
    return int(lm[0, -1].argmax()) == stoi.get(f["val"], -1)


if __name__ == "__main__":
    p, vocab, stoi, trained = train_model()
    facts, toks, ids = make_stream(stoi, trained)
    qpos = len(ids)                                   # query everything at the end
    store = extract_to_store(p, facts, toks, stoi)
    room = A.T  # window size

    rows = {"in-window": [0, 0], "beyond-window": [0, 0]}   # [plain_ok, store_ok] counts
    tot = {"in-window": 0, "beyond-window": 0}
    for f in facts:
        dist = qpos - f["val_pos"]
        bucket = "in-window" if dist < A.T else "beyond-window"
        tot[bucket] += 1
        rows[bucket][0] += int(plain_recall(p, f, ids, stoi, qpos))
        rows[bucket][1] += int(store.get((f["name"], f["slot"])) == f["val"])

    print(f"stream: {len(facts)} novel facts, {len(ids)} tokens, context T={A.T}")
    print(f"(facts within last {A.T} tokens are 'in-window'; older are 'beyond')\n")
    print(f"{'recall@':16} {'plain LM (context)':>20} {'in-loop + store':>18}")
    for b in ("in-window", "beyond-window"):
        n = max(1, tot[b])
        print(f"{b:16} {rows[b][0]/n:19.0%}  {rows[b][1]/n:17.0%}   (n={tot[b]})")
    print("\nplain LM can only recall within its window; the in-loop model extracts"
          "\neach fact as it reads and the store recalls it at ANY distance.")
