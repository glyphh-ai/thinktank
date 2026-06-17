"""
WHY — read the model's pick back out as meaning.
=================================================
The model trains normally (weights only). When it picks the next glyph, we don't
change anything about generation -- we look up the FIXED meaning vector that sits
in that same vocab slot (the identically aligned glyph) and decode it to explain
WHY, in human terms: was the pick audible (a vowel), a stop, a closed-loop shape?

This is memory + introspection, not generation: the meaning store is fixed; we
align the decision to it and unbind the role paths (semantic->linguistic->sound,
semantic->physical->shape) to surface the dominant features behind the choice.

Run:  python why.py
"""

import numpy as np
import step3_gpt as G
import step1_data as D
import vsa

# feature atoms we score the decision against (the readable vocabulary of "why")
SOUND_FEATS = ["vowel", "consonant", "voiced", "unvoiced", "plosive", "fricative",
               "nasal", "liquid", "approximant", "open", "close", "front", "back",
               "silence", "separator", "stop", "pause"]
SHAPE_FEATS = ["closed-loop", "open-curve", "ascender", "descender", "x-height",
               "vertical-stem", "horizontal-bar", "diagonal", "arch", "dot",
               "curve", "symmetric", "whitespace", "punctuation"]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def unbind_path(v, keys, sp):
    """Walk the role-key chain on a (real-valued) vector to expose a leaf."""
    out = v.astype(np.float64)
    for k in keys:
        out = out * sp.atom(k)            # unbind = multiply (works on real sums too)
    return out


def explain(p, context, sp, glyph, topk=3):
    ids = [D.stoi[c] for c in context][-G.BLOCK:]
    logits, _ = G.forward(p, np.array([ids]))
    z = logits[0, -1]
    probs = np.exp(z - z.max()); probs /= probs.sum()

    # expected meaning of the DECISION: superpose the aligned glyph vectors,
    # weighted by how much the model wanted each one.
    V = np.zeros(vsa.D)
    for i, pr in enumerate(probs):
        V += pr * glyph[D.itos[i]].astype(np.float64)

    snd = unbind_path(V, ["semantic", "linguistic", "sound"], sp)
    shp = unbind_path(V, ["semantic", "physical", "shape"], sp)
    sound = sorted(((f, cos(snd, sp.atom(f))) for f in SOUND_FEATS), key=lambda t: -t[1])
    shape = sorted(((f, cos(shp, sp.atom(f))) for f in SHAPE_FEATS), key=lambda t: -t[1])
    top = sorted(range(len(probs)), key=lambda i: -probs[i])[:topk]
    return probs, top, sound, shape


def main():
    G.set_frozen_tok(None)                      # weights only; meaning is read-out, not input
    print("training baseline (weights only)...")
    p, _ = G.train(steps=800, seed=0, verbose=False, eval_every=800)

    sp = vsa.HDSpace(D=vsa.D, seed=0)
    glyph = vsa.build_vocab_vectors(D.chars, sp)   # id-aligned fixed meaning memory

    print("\nReading each pick back out as meaning (the 'why'):\n")
    for ctx in ["to b", "th", "and ", "questio", "or to ", "them."]:
        probs, top, sound, shape = explain(p, ctx, sp, glyph)
        picks = "  ".join(f"{D.itos[i]!r}={probs[i]:.2f}" for i in top)
        print(f"after {ctx!r:10} -> picks: {picks}")
        print("   why (sound): " + ", ".join(f"{f} {s:+.2f}" for f, s in sound[:3]))
        print("   why (shape): " + ", ".join(f"{f} {s:+.2f}" for f, s in shape[:3]))
        # explicit audible-vs-silent read of the decision
        aud = cos(unbind_path(sum(probs[i]*glyph[D.itos[i]].astype(float)
                                  for i in range(len(probs))),
                              ["semantic", "linguistic", "sound"], sp), sp.atom("vowel"))
        print(f"   audible(vowel-ness of the pick): {aud:+.2f}\n")


if __name__ == "__main__":
    main()
