"""
vsa.py — JSON-as-letters: meaning encoded as hypervectors.
=========================================================
A Vector Symbolic Architecture (VSA / hyperdimensional computing) that encodes
a JSON-like structure of meaning into one fixed-width vector, exactly per spec:

    Role    -> key/value BOUND vector           role  = bind(key, value)
    Segment -> bind/bundle of all role vectors  seg   = bind(name, bundle(roles))
    Layer   -> bind/bundle of all segments      layer = bind(name, bundle(segs))
    Cortex  -> bind/bundle of all layers         cortex= bundle(layers)

So any structure can be described and compared with math:

    Cortex -> semantic -> physical   -> shape: <letter-shape>
    Cortex -> semantic -> linguistic -> sound: <letter-sound>

Each node may carry a WEIGHT (used when bundling siblings) so it carries signal.
Two encoded things are compared by cosine similarity (dense) or Hamming distance
(bipolar) — for bipolar vectors the two are linearly related.

Atoms are grounded in a prime basis (NSM semantic primes + feature primitives),
each a near-orthogonal random bipolar vector, so distinct meanings stay separated
while shared features pull things together.

Choices (spec left these open): bipolar {-1,+1}, D=2048, bind=elementwise mult
(self-inverse), bundle=weighted sign(sum).

Run:  python vsa.py
"""

import numpy as np

D = 2048  # hypervector width


# ============================================================================
# Hyperdimensional space: atoms + the two core ops (bind, bundle)
# ============================================================================
class HDSpace:
    """Holds the named atomic hypervectors. Each name -> one fixed random
    bipolar vector, near-orthogonal to every other (the prime basis)."""
    def __init__(self, D=D, seed=0):
        self.D = D
        self._seed = seed
        self._atoms = {}

    def atom(self, name):
        """Deterministic random bipolar hypervector for `name` (memoized)."""
        if name not in self._atoms:
            # seed per-name so atoms are reproducible and independent
            h = (hash((self._seed, name)) & 0xFFFFFFFF)
            rng = np.random.default_rng(h)
            self._atoms[name] = rng.choice(np.array([-1, 1]), size=self.D)
        return self._atoms[name]


def bind(a, b):
    """Associate two vectors (role <- key x value). Elementwise multiply:
    result is dissimilar to both inputs, and self-inverse: bind(bind(a,b),b)=a."""
    return a * b


def unbind(a, b):
    """Recover: unbind(bind(k,v), k) == v   (multiply is its own inverse)."""
    return a * b


def bundle(vectors, weights=None):
    """Superpose into one vector that is SIMILAR to all inputs (the set/record).
    Weighted majority: sign(sum_i w_i v_i). Ties -> +1."""
    V = np.asarray(vectors, dtype=np.float64)
    if weights is None:
        s = V.sum(0)
    else:
        s = (np.asarray(weights)[:, None] * V).sum(0)
    return np.where(s >= 0, 1, -1).astype(np.int8)


# ---- similarity --------------------------------------------------------------
def cosine(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def hamming(a, b):
    """Number of differing positions (bipolar)."""
    return int((a != b).sum())

def hamming_sim(a, b):
    """1 - normalized Hamming = fraction of agreeing bits in [0,1]."""
    return 1.0 - hamming(a, b) / len(a)


# ============================================================================
# Encode a JSON-like tree of meaning into one Cortex vector
# ============================================================================
def encode(node, space):
    """Recursively encode a nested dict into a hypervector.

    - leaf str         -> the atom for that name
    - leaf list[str]   -> bundle of the atoms for those feature names
    - leaf np.ndarray  -> used as-is
    - dict {k: child}  -> bundle over k of bind(atom(k), encode(child))
      (each key binds exactly once, so a path is recoverable by unbinding the
       key chain). An optional "_w" dict weights the siblings in the bundle.
    """
    if isinstance(node, np.ndarray):
        return node
    if isinstance(node, str):
        return space.atom(node)
    if isinstance(node, (list, tuple)):
        return bundle([space.atom(s) for s in node])
    if isinstance(node, dict):
        weights = node.get("_w", {})
        children, ws = [], []
        for k, child in node.items():
            if k == "_w":
                continue
            children.append(bind(space.atom(k), encode(child, space)))
            ws.append(weights.get(k, 1.0))
        return bundle(children, ws)
    raise TypeError(f"cannot encode {type(node)}")


def query(cortex, key_chain, space):
    """Walk a path by unbinding the key chain; returns the recovered vector.
    e.g. query(cortex, ['semantic','linguistic','sound'], space)."""
    v = cortex
    for k in key_chain:
        v = unbind(v, space.atom(k))
    return v


# ============================================================================
# The letter schema: physical (shape) + linguistic (sound) feature primes
# ============================================================================
# NSM (Natural Semantic Metalanguage) primes — the grounding inventory. Letters
# don't carry rich NSM semantics individually, but the primes seed the atom
# space as the universal basis the schema can reference.
NSM_PRIMES = [
    "I","YOU","SOMEONE","SOMETHING","PEOPLE","BODY","KIND","PART","THIS",
    "SAME","OTHER","ONE","TWO","SOME","ALL","MANY","LITTLE","GOOD","BAD",
    "BIG","SMALL","THINK","KNOW","WANT","FEEL","SEE","HEAR","SAY","WORD",
    "TRUE","DO","HAPPEN","MOVE","BE","LIVE","DIE","WHEN","NOW","BEFORE",
    "AFTER","WHERE","HERE","ABOVE","BELOW","NEAR","FAR","SIDE","INSIDE",
    "TOUCH","NOT","MAYBE","CAN","BECAUSE","IF","VERY","MORE","LIKE",
]

# Linguistic feature primes per lowercase letter (primary phoneme).
SOUND = {
    "a": ["vowel","open","front"],          "b": ["consonant","voiced","plosive","labial"],
    "c": ["consonant","unvoiced","plosive","velar"], "d": ["consonant","voiced","plosive","alveolar"],
    "e": ["vowel","mid","front"],           "f": ["consonant","unvoiced","fricative","labiodental"],
    "g": ["consonant","voiced","plosive","velar"],   "h": ["consonant","unvoiced","fricative","glottal"],
    "i": ["vowel","close","front"],         "j": ["consonant","voiced","affricate","postalveolar"],
    "k": ["consonant","unvoiced","plosive","velar"], "l": ["consonant","voiced","liquid","alveolar"],
    "m": ["consonant","voiced","nasal","labial"],    "n": ["consonant","voiced","nasal","alveolar"],
    "o": ["vowel","mid","back","round"],    "p": ["consonant","unvoiced","plosive","labial"],
    "q": ["consonant","unvoiced","plosive","velar"], "r": ["consonant","voiced","liquid","alveolar"],
    "s": ["consonant","unvoiced","fricative","alveolar"], "t": ["consonant","unvoiced","plosive","alveolar"],
    "u": ["vowel","close","back","round"],  "v": ["consonant","voiced","fricative","labiodental"],
    "w": ["consonant","voiced","approximant","labial"], "x": ["consonant","unvoiced","fricative","velar"],
    "y": ["consonant","voiced","approximant","palatal"], "z": ["consonant","voiced","fricative","alveolar"],
}

# Physical (lowercase glyph) shape feature primes.
SHAPE = {
    "a": ["closed-loop","x-height","curve"],   "b": ["closed-loop","ascender","vertical-stem"],
    "c": ["open-curve","x-height"],            "d": ["closed-loop","ascender","vertical-stem"],
    "e": ["closed-loop","x-height","horizontal-bar"], "f": ["ascender","vertical-stem","horizontal-bar"],
    "g": ["closed-loop","descender","curve"],  "h": ["ascender","vertical-stem","arch"],
    "i": ["x-height","vertical-stem","dot"],   "j": ["descender","vertical-stem","dot"],
    "k": ["ascender","vertical-stem","diagonal"], "l": ["ascender","vertical-stem"],
    "m": ["x-height","vertical-stem","arch"],  "n": ["x-height","vertical-stem","arch"],
    "o": ["closed-loop","x-height","curve","symmetric"], "p": ["closed-loop","descender","vertical-stem"],
    "q": ["closed-loop","descender","vertical-stem"], "r": ["x-height","vertical-stem","arch"],
    "s": ["open-curve","x-height","symmetric"],"t": ["ascender","vertical-stem","horizontal-bar"],
    "u": ["x-height","curve","arch"],          "v": ["x-height","diagonal","symmetric"],
    "w": ["x-height","diagonal"],              "x": ["x-height","diagonal","symmetric"],
    "y": ["descender","diagonal"],             "z": ["x-height","diagonal","horizontal-bar"],
}


# Non-letter glyphs (whitespace / punctuation) so EVERY vocab char has meaning.
SOUND.update({
    " ":  ["silence","separator"],      "\n": ["silence","boundary"],
    ",":  ["pause","short"],            ".":  ["stop","final"],
    ":":  ["introduce","pause"],
})
SHAPE.update({
    " ":  ["whitespace","gap"],         "\n": ["whitespace","line-break"],
    ",":  ["punctuation","low-mark","curve"], ".": ["punctuation","dot","low-mark"],
    ":":  ["punctuation","two-dots"],
})


def letter_cortex(ch, space, w_shape=1.0, w_sound=1.0):
    """Build the Cortex vector for one letter per the example schema:
        cortex -> semantic -> { physical -> shape:<feats>,
                                linguistic -> sound:<feats> }
    Weights let shape vs sound carry different signal."""
    tree = {
        "semantic": {
            "physical":   {"shape": SHAPE[ch]},
            "linguistic": {"sound": SOUND[ch]},
            "_w": {"physical": w_shape, "linguistic": w_sound},
        }
    }
    return encode(tree, space)


def build_alphabet(space, **kw):
    return {ch: letter_cortex(ch, space, **kw) for ch in SHAPE}


def build_vocab_vectors(chars, space, **kw):
    """Cortex vector for every char in `chars` (model vocab order). Chars not in
    the schema fall back to their own near-orthogonal atom."""
    out = {}
    for ch in chars:
        if ch in SHAPE and ch in SOUND:
            out[ch] = letter_cortex(ch, space, **kw)
        else:
            out[ch] = space.atom(f"glyph:{ch!r}")
    return out


def project_to_embedding(vectors_by_char, C, seed=0):
    """Project the D-dim meaning vectors down to the model's embedding dim C
    with a fixed random Gaussian projection -> a (vocab, C) matrix usable as the
    token-embedding init in the training loop ('use the glyph, not the char')."""
    chars = sorted(vectors_by_char)
    M = np.stack([vectors_by_char[c].astype(np.float64) for c in chars])  # (V, D)
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1/np.sqrt(len(M[0])), size=(M.shape[1], C))          # (D, C)
    E = M @ P                                                              # (V, C)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return chars, E


# ============================================================================
# Self-test / demo
# ============================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    sp = HDSpace(D=D, seed=0)

    print(f"D = {D}   atoms are near-orthogonal:")
    a, b, c = sp.atom("vowel"), sp.atom("plosive"), sp.atom("labial")
    print(f"  cos(vowel, plosive) = {cosine(a,b):+.3f}   "
          f"cos(vowel, labial) = {cosine(a,c):+.3f}   (≈0 expected, 1/√D={1/np.sqrt(D):.3f})")

    print("\nbind is self-inverse, bundle keeps similarity:")
    k, v = sp.atom("shape"), bundle([sp.atom("closed-loop"), sp.atom("curve")])
    r = bind(k, v)
    print(f"  cos(role, key)   = {cosine(r,k):+.3f}  (bound -> dissimilar to key)")
    print(f"  cos(unbind, val) = {cosine(unbind(r,k),v):+.3f}  (recovered the value)")

    alpha = build_alphabet(sp)

    print("\nseparation of the 26 letter Cortex vectors:")
    chars = sorted(alpha)
    Mtx = np.stack([alpha[c] for c in chars]).astype(np.float64)
    Cn = Mtx / np.linalg.norm(Mtx, axis=1, keepdims=True)
    S = Cn @ Cn.T
    off = S[~np.eye(26, dtype=bool)]
    print(f"  mean off-diagonal cos = {off.mean():+.3f}   max = {off.max():+.3f}  "
          f"(low => well separated)")

    def sim(x, y): return cosine(alpha[x], alpha[y])
    print("\nstructure shows up in similarity:")
    print(f"  vowels   cos(a,e)={sim('a','e'):+.3f}  cos(i,u)={sim('i','u'):+.3f}")
    print(f"  loop fam cos(b,d)={sim('b','d'):+.3f}  cos(p,q)={sim('p','q'):+.3f}")
    print(f"  cross    cos(a,k)={sim('a','k'):+.3f}  cos(o,t)={sim('o','t'):+.3f}")

    print("\nquery a path back out of the Cortex (unbind the key chain):")
    rec = query(alpha["a"], ["semantic","linguistic","sound"], sp)
    want = bundle([sp.atom(s) for s in SOUND["a"]])
    # the recovered leaf should resemble 'a's sound descriptor more than a wrong one
    wrong = bundle([sp.atom(s) for s in SOUND["t"]])
    print(f"  cos(recovered, a-sound)={cosine(rec,want):+.3f}  vs  "
          f"cos(recovered, t-sound)={cosine(rec,wrong):+.3f}")

    chars, E = project_to_embedding(alpha, C=64)
    print(f"\nprojected to (vocab={len(chars)}, C={E.shape[1]}) embedding init for the loop.")
