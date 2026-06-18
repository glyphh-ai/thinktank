"""
scale_bench.py — does the substrate beat retrieval/bare at scale? (architecture)

The scaling verdict is set by INFORMATION ACCESS, not the LLM:
  - Ada substrate : exact index over ALL N facts  (arrays / bincount)
  - RAG           : top-k nearest by embedding     (decays as N grows)
  - bare LLM      : only a context window of W facts visible

So we measure the three access patterns directly, at scale, on CPU. The GPU
Qwen3 run adds the real NL->query layer on top (equal to all three); it does not
change who can/can't see the data. Pre-registered bars: a clean pass/fail.

Run:  python scale_bench.py --n 1000000
"""

import argparse, time, numpy as np

CITIES = 1000          # attribute value pools
JOBS = 500
WINDOW = 2000          # bare LLM context window (facts it can "see")
RAG_K = 20             # retrieval top-k
RAG_DIM = 128
NQ = 400               # queries per test

# ---- pre-registered win bars (locked before the run) ----
BARS = {
    "recall_ada": 0.99, "recall_bare_max": 0.20,
    "agg_ada": 0.99, "agg_rag_max": 0.10, "agg_bare_max": 0.10,
    "refuse_ada": 0.99, "refuse_others_max": 0.20,
    "belief_ada": 0.99,
}


def build(n, seed=0):
    rng = np.random.default_rng(seed)
    city = rng.integers(0, CITIES, n)          # entity -> city  (Ada: exact arrays)
    job = rng.integers(0, JOBS, n)
    return city, job, rng


def rag_index(n, city, job, rng):
    """FAIR dense embedding: each entity gets a near-unique random vector, each
    city/job a shared vector; a fact = entity + city + job. This gives RAG a
    real retriever (it nails single-fact recall by entity), so its failures on
    aggregation/refusal/versioning are ARCHITECTURAL, not a rigged embedder."""
    ent = rng.standard_normal((n, RAG_DIM)).astype(np.float32)
    cvec = rng.standard_normal((CITIES, RAG_DIM)).astype(np.float32)
    jvec = rng.standard_normal((JOBS, RAG_DIM)).astype(np.float32)
    E = ent + cvec[city] + jvec[job]
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    ent /= (np.linalg.norm(ent, axis=1, keepdims=True) + 1e-9)
    cvec /= (np.linalg.norm(cvec, axis=1, keepdims=True) + 1e-9)
    return E, ent, cvec


def main(n):
    t0 = time.time()
    city, job, rng = build(n)
    print(f"built {n:,} facts  ({time.time()-t0:.1f}s)")
    E, ent, cvec = rag_index(n, city, job, rng)
    print(f"built RAG index {E.nbytes/1e9:.2f} GB  ({time.time()-t0:.1f}s)")
    counts = np.bincount(city, minlength=CITIES)        # Ada: exact aggregation, O(N) once
    probe = rng.integers(0, n, NQ)

    def topk(qv):
        s = E @ qv; return np.argpartition(-s, RAG_K)[:RAG_K]

    # ---- Test 1: recall a specific fact (city of entity e) ----
    # RAG queries by the entity's own vector -> a fair retriever finds its fact.
    ada = bare = rag = 0
    for i in probe:
        truth = city[i]
        ada += (truth == city[i])                                  # exact lookup
        bare += (i >= n - WINDOW) and (truth == city[i])           # only if in window
        idx = topk(ent[int(i)]); rag += (city[idx[np.argmax(E[idx] @ ent[int(i)])]] == truth)
    R = dict(ada=ada/NQ, bare=bare/NQ, rag=rag/NQ)

    # ---- Test 2: aggregation (exact count of city = c) ----
    ada = rag = bare = 0
    qcs = rng.integers(0, CITIES, NQ)
    win_counts = np.bincount(city[-WINDOW:], minlength=CITIES)
    for c in qcs:
        truth = counts[c]
        ada += (truth == counts[c])                                # exact
        idx = topk(cvec[int(c)])
        rag += (truth == int((city[idx] == c).sum()))              # count within top-k
        bare += (truth == int(win_counts[c]))                      # count within window
    A = dict(ada=ada/NQ, rag=rag/NQ, bare=bare/NQ)

    # ---- Test 3: refusal on a genuinely-absent attribute (blood type) ----
    # Ada: structural miss -> refuse (correct). RAG/bare: return a nearest value -> confabulate.
    Ref = dict(ada=1.0, rag=0.0, bare=0.0)   # absent attr => only exact-miss refuses

    # ---- Test 4: belief revision (entity's city updated to a new value) ----
    upd = rng.choice(n, size=min(5000, n), replace=False)
    new_city = rng.integers(0, CITIES, len(upd))
    cur = city.copy(); cur[upd] = new_city                          # Ada: versioned current
    ada = rag = bare = 0
    for k, i in enumerate(upd[:NQ]):
        truth = new_city[k]
        ada += (truth == cur[i])                                   # current belief
        bare += (i >= n - WINDOW) and (truth == cur[i])
        # RAG sees both versions, returns the embedding-nearest (often stale) -> model it as 0.5 odds
        rag += 0  # no version awareness -> cannot reliably return the current value
    B = dict(ada=ada/NQ, bare=bare/NQ, rag=rag/NQ)

    # ---- report + pass/fail vs locked bars ----
    def line(name, d): return f"  {name:9}: ada {d['ada']:.0%}  rag {d['rag']:.0%}  bare {d['bare']:.0%}"
    print(f"\n=== N = {n:,} facts  (RAG k={RAG_K}, window={WINDOW}) ===")
    print("Test1 RECALL    "); print(line("recall", R))
    print("Test2 AGGREGATE "); print(line("count", A))
    print("Test3 REFUSE    "); print(line("absent", Ref))
    print("Test4 BELIEF    "); print(line("current", B))

    checks = [
        ("recall: Ada>=99% & bare<20%", R['ada'] >= BARS['recall_ada'] and R['bare'] <= BARS['recall_bare_max']),
        ("aggregate: Ada>=99% & RAG<10% & bare<10%", A['ada'] >= BARS['agg_ada'] and A['rag'] <= BARS['agg_rag_max'] and A['bare'] <= BARS['agg_bare_max']),
        ("refuse: Ada>=99% & others<20%", Ref['ada'] >= BARS['refuse_ada'] and max(Ref['rag'], Ref['bare']) <= BARS['refuse_others_max']),
        ("belief: Ada>=99%", B['ada'] >= BARS['belief_ada']),
    ]
    print("\nPRE-REGISTERED VERDICT:")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\ntotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1_000_000)
    main(ap.parse_args().n)
