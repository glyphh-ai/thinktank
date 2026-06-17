"""
Use the glyph MEANING vectors during weights instead of the generic char.

Pipeline:
  1. encode every vocab char into a Cortex meaning-vector (vsa.py)
  2. project D -> C  -> a (vocab, C) token-embedding matrix
  3. seed the GPT's tok_emb with it and train
  4. after training -> 'meaning as weighted vector': the embeddings started as
     structured meaning and were then tuned by the loss.

Compares VSA-seeded init vs the usual random init (same seed/data/budget).

Run:  python train_meaning.py
"""

import numpy as np
import step3_gpt as G
import step1_data as D
import vsa


def vsa_embedding(C, proj_seed=0):
    """(vocab, C) matrix of projected glyph meaning-vectors, in token-id order
    (= sorted(chars), which is exactly step1_data's stoi ordering)."""
    sp = vsa.HDSpace(D=vsa.D, seed=0)
    vecs = vsa.build_vocab_vectors(D.chars, sp)
    chars, E = vsa.project_to_embedding(vecs, C=C, seed=proj_seed)
    assert chars == sorted(D.chars) == [D.itos[i] for i in range(len(chars))]
    # scale to match the usual 0.02 init magnitude so lr/dynamics stay comparable
    E = E / E.std() * 0.02
    return E


def steps_to(hist, target=0.90):
    for step, _, _, _, va in hist:
        if va >= target:
            return step
    return None


def run(tok_emb_init, seed, steps, eval_every):
    p, hist = G.train(steps=steps, B=32, seed=seed, tok_emb_init=tok_emb_init,
                      eval_every=eval_every, verbose=False)
    eval_rng = np.random.default_rng(123)
    va = [G.evaluate(p, "val", eval_rng) for _ in range(8)]
    return p, hist, np.mean([l for l, _ in va]), np.mean([a for _, a in va])


def main(steps=600, seeds=(0, 1, 2), eval_every=25):
    C = G.C
    E = vsa_embedding(C)
    print(f"vocab={G.vocab_size} C={C} T={G.BLOCK} | {steps} steps, seeds {seeds}")
    print(f"VSA token-embedding init: {E.shape}\n")

    # --- structure carried into the embedding init (cosine in C-dim) ---
    def ecos(a, b):
        u, v = E[D.stoi[a]], E[D.stoi[b]]
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    print("init embedding structure (projected to C): "
          f"cos(b,d)={ecos('b','d'):+.2f}  cos(p,q)={ecos('p','q'):+.2f}  "
          f"cos(a,e)={ecos('a','e'):+.2f}  cos(a,k)={ecos('a','k'):+.2f}\n")

    res = {}
    for name, init in (("random", None), ("VSA-meaning", E)):
        hists, vls, vas, hit = [], [], [], []
        last_p = None
        for s in seeds:
            p, h, vl, va = run(init, s, steps, eval_every)
            hists.append(h); vls.append(vl); vas.append(va)
            st = steps_to(h, 0.90);
            if st: hit.append(st)
            last_p = p
        res[name] = dict(hists=hists, vl=vls, va=vas, hit=hit, p=last_p)

    print(f"{'init':14} {'val loss(mean±std)':>20} {'val acc':>9} {'->90% @':>9}")
    for name in ("random", "VSA-meaning"):
        r = res[name]
        h = f"{np.mean(r['hit']):.0f}" if r["hit"] else "never"
        print(f"{name:14} {np.mean(r['vl']):10.3f} ± {np.std(r['vl']):5.3f}   "
              f"{np.mean(r['va']):8.1%} {h:>9}")

    # --- 'meaning as weighted vector': structure AFTER training (seed 0) ---
    tok = res["VSA-meaning"]["p"]["tok_emb"]
    def tcos(a, b):
        u, v = tok[D.stoi[a]], tok[D.stoi[b]]
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    print("\ntrained embedding (meaning as weighted vector): "
          f"cos(b,d)={tcos('b','d'):+.2f}  cos(p,q)={tcos('p','q'):+.2f}  "
          f"cos(a,e)={tcos('a','e'):+.2f}  cos(a,k)={tcos('a','k'):+.2f}")


if __name__ == "__main__":
    main()
