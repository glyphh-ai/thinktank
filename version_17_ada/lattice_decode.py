"""
LATTICE DECODE — meaning as energy spreading through the lattice
================================================================
Decoding is not one function (unbind once, nearest word). A word FIRES:

  1. its meaning vector m(w) is probed down every thread of the lattice,
     layer -> segment -> role:   x = m(w) ⊙ K(layer) ⊙ K(segment) ⊙ K(role)
  2. each probe is cleaned up against every glyph; a thread FIRES when its
     best match clears the noise floor (measured on roles the word leaves
     empty), landing on a value word
  3. the word's energy splits across the threads that fired, in proportion
     to their match strength, and decays by β per hop
  4. every value word reached fires in turn with the energy it received —
     the walk spreads through the closed set until energy falls below ε

What is left is an ACTIVATION FIELD: energy on every lattice node (layer,
segment, role) and on every word reached, plus the threads that carried it.
That field is the decoded meaning — the same structure as the input.

FIXED LATTICE (default): every glyph binds all 27 roles — a role a word
leaves empty binds the word "none", itself a glyph (absence, zero, opposite
of all) — so every glyph has the identical structure and the same energy per
thread. --free binds only the filled roles (glyphs differ in shape).

Two words are compared node by node: per layer, the cosine of the word
fields that entered through that layer — and the threads that run directly
from one word to the other (love -opposite-> hate).

Glyph sets: glyphs3.jsonl (v3, default) — IDENTICAL atomic makeup: every
glyph is 27 real words, unweighted (27 binds x 1/27). glyphs2.jsonl (v2)
has per-role weights and empty roles.

Run:  python lattice_decode.py                       # love hate kind mean
      python lattice_decode.py love hate --depth 3
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import open_encode2 as E
from glyph_schema2 import SCHEMA, ROLE_PATHS, PATHS

HERE = Path(__file__).parent
LAYERS = list(SCHEMA)


# ============================================================================
# The space: glyphs + glyph keys (solved once, cached)
# ============================================================================
class Lattice:
    def __init__(self, beta_glyph=0.7, rebuild=False, fixed=True, w_none=0.3,
                 glyphs="glyphs2.jsonl"):
        G = E.load(glyphs, fixed=fixed, w_none=w_none)
        if fixed:
            assert all(len(t) == len(PATHS) for t in G.values()), "fixed lattice: 27 binds each"
        self.space = space = E.Space(G)
        stem = Path(glyphs).stem
        CACHE = HERE / (f"space_{stem}_fixed_{w_none}.npz" if fixed else f"space_{stem}.npz")
        if CACHE.exists() and not rebuild:
            z = np.load(CACHE, allow_pickle=True)
            g, K = z["g"], z["K"].item()
            M = dict(zip(z["mwords"], z["M"]))
        else:
            _, M0, _ = space.solve(E.keys_from_atoms(), beta_glyph)
            K = E.keys_from_meanings(M0)                     # glyph keys (one step)
            g, M, _ = space.solve(K, beta_glyph)
            np.savez(CACHE, g=g, K=np.array(K, dtype=object),
                     mwords=np.array(list(M)), M=np.stack(list(M.values())))
        self.K, self.M = K, M
        self.names = space.names
        self.mem = g / np.linalg.norm(g, axis=1, keepdims=True)   # cleanup memory
        self.Kmat = np.stack([K[p] for p in PATHS])                # (27, D)
        self._fired = {}
        self.floor = self._noise_floor()

    def _noise_floor(self, sample=300, n_keys=27, q=99.5):
        """Cleanup score of probes with random keys that belong to no role = pure noise.
        (In a fixed lattice every role is filled, so there are no empty roles to use.)"""
        rng = np.random.default_rng(0)
        words = rng.choice(self.space.words, size=min(sample, len(self.space.words)),
                           replace=False)
        rand_keys = rng.choice(np.array([-1, 1], np.float32), (n_keys, self.Kmat.shape[1]))
        noise = []
        for w in words:
            scores, _ = self._probe(w, rand_keys)
            noise += list(scores)
        return float(np.percentile(noise, q))

    def _probe(self, w, keys=None):
        keys = self.Kmat if keys is None else keys
        m = self.M[w] / np.linalg.norm(self.M[w])
        sims = self.mem @ (keys * m).T                             # (N, n_keys)
        best = sims.argmax(0)
        return sims[best, np.arange(keys.shape[0])], best

    def fire(self, w):
        """One neuron firing: [(role_path, value_word, strength)] above the floor."""
        if w not in self._fired:
            if w not in self.M:                                    # a leaf: no glyph, absorbs
                self._fired[w] = []
            else:
                scores, best = self._probe(w)
                self._fired[w] = [(p, self.names[int(b)], float(s))
                                  for p, s, b in zip(PATHS, scores, best)
                                  if s > self.floor and self.names[int(b)] != w]
        return self._fired[w]

    # ------------------------------------------------------------------------
    def spread(self, w, depth=3, beta=0.5, eps=1e-5, branch=None):
        """Energy from w down the lattice threads, hop by hop.

        Returns a field: node energy (role paths), word energy split by the
        LAYER it entered through at hop 1, and the threads that fired.
        """
        node = defaultdict(float)
        word_by_layer = {l: defaultdict(float) for l in LAYERS}
        threads = []
        frontier = [(w, 1.0, None, 0)]                             # (word, energy, entry layer, hop)
        while frontier:
            nxt = []
            for src, energy, entry, hop in frontier:
                fired = sorted(self.fire(src), key=lambda t: -t[2])[:branch]
                total = sum(s for _, _, s in fired)
                for path, val, s in fired:
                    e = energy * beta * s / total
                    if e < eps:
                        continue
                    layer = path.split(".")[0]
                    lay = entry or layer
                    node[path] += e
                    word_by_layer[lay][val] += e
                    threads.append((hop + 1, src, path, val, e))
                    if hop + 1 < depth:
                        nxt.append((val, e, lay, hop + 1))
            frontier = nxt
        return {"node": dict(node), "words": word_by_layer, "threads": threads}


# ============================================================================
# Reading the field
# ============================================================================
def show(L, w, field):
    print(f"\n{w}  — first hop, the lattice that fired (noise floor {L.floor:.3f})")
    first = [t for t in field["threads"] if t[0] == 1]
    seg_e = defaultdict(float)
    for _, _, path, _, e in first:
        seg_e[".".join(path.split(".")[:2])] += e
    for layer in LAYERS:
        le = sum(e for k, e in seg_e.items() if k.startswith(layer + "."))
        if le == 0:
            continue
        print(f"  {layer:9} {le:.3f}")
        for seg in SCHEMA[layer]:
            se = seg_e.get(f"{layer}.{seg}", 0)
            if se == 0:
                continue
            print(f"     {seg:11} {se:.3f}")
            for _, _, path, val, e in first:
                if path.startswith(f"{layer}.{seg}."):
                    print(f"        {path.split('.')[-1]:9} -> {val:12} {e:.3f}")
    reached = defaultdict(float)
    for l in LAYERS:
        for v, e in field["words"][l].items():
            reached[v] += e
    top = sorted(reached.items(), key=lambda t: -t[1])[:12]
    print(f"  spread: {len(reached)} words reached, {len(field['threads'])} threads fired; strongest:")
    print("     " + ", ".join(f"{v} {e:.3f}" for v, e in top))


def field_cos(a: dict, b: dict) -> float:
    keys = a.keys() | b.keys()
    va = np.array([a.get(k, 0) for k in keys]); vb = np.array([b.get(k, 0) for k in keys])
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return float(va @ vb / (na * nb)) if na and nb else 0.0


def compare(L, a, b, fa, fb):
    print(f"\n{a} vs {b}")
    for layer in LAYERS:
        print(f"  {layer:9} {field_cos(fa['words'][layer], fb['words'][layer]):5.2f}")
    print(f"  lattice   {field_cos(fa['node'], fb['node']):5.2f}   (which threads carry energy)")
    direct = [(h, s, p, v, e) for h, s, p, v, e in fa["threads"] if v == b] + \
             [(h, s, p, v, e) for h, s, p, v, e in fb["threads"] if v == a]
    for h, s, p, v, e in sorted(direct, key=lambda t: (t[0], -t[4]))[:6]:
        print(f"  thread    {s} -{p.split('.')[-1]}-> {v}   (hop {h}, energy {e:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("words", nargs="*", default=["love", "hate", "kind", "mean"])
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--beta", type=float, default=0.5, help="energy kept per hop")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--free", action="store_true", help="glyphs bind only the roles they fill")
    ap.add_argument("--w-none", type=float, default=0.3, help="weight of a 'none' bind (fixed)")
    ap.add_argument("--glyphs", default="glyphs3.jsonl",
                    help="glyphs3.jsonl = v3, identical makeup (default); glyphs2.jsonl = v2")
    args = ap.parse_args()

    L = Lattice(rebuild=args.rebuild, fixed=not args.free, w_none=args.w_none, glyphs=args.glyphs)
    fields = {w: L.spread(w, depth=args.depth, beta=args.beta) for w in args.words}
    for w in args.words:
        show(L, w, fields[w])
    ws = args.words
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            compare(L, ws[i], ws[j], fields[ws[i]], fields[ws[j]])
