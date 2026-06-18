"""
prepare_data.py — turn adaL's real enriched facts into training + benchmark data
for the Qwen3 frontier test. Validatable on CPU; no GPU, no API key.

Outputs (under ./data/):
  facts.jsonl : {text, word_roles}  -- NL fact + per-word Ada role (Arm B labels)
  qa.jsonl    : {question, answer, entity, slot}  -- QA over the facts (both arms)

Run:  python prepare_data.py --n 50000
"""

import os, json, argparse, subprocess, random

ADAL = "/tmp/adaL"
SRC = f"{ADAL}/benchmark/curriculum/agi/ada.jsonl"

# Ada universal schema (vendored from glyphh-ai/adaL, Apache-2.0)
UNIVERSAL_SCHEMA = {
    "entity": ["name", "kind", "subkind"],
    "perceptual": ["color", "size", "shape", "texture", "sound", "smell", "taste", "temperature"],
    "spatial": ["location", "origin", "direction"],
    "temporal": ["time", "duration", "age", "era", "frequency"],
    "relational": ["subject", "predicate", "object", "possessor", "agent", "patient", "instrument"],
    "quantitative": ["count", "magnitude", "unit", "ratio"],
    "epistemic": ["source", "certainty", "modality"],
}
ALL_ROLES = [f"{l}.{r}" for l, rs in UNIVERSAL_SCHEMA.items() for r in rs]

# slot -> natural-language question template
QUESTION = {
    "relational.object":  "What does {e} relate to?",
    "relational.predicate": "What does {e} do?",
    "entity.kind":        "What kind of thing is {e}?",
    "entity.subkind":     "What subtype is {e}?",
    "spatial.location":   "Where is {e}?",
}


def ensure():
    if not os.path.exists(SRC):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/glyphh-ai/adaL", ADAL],
                       check=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})


def words(v, cap=4):
    return [t for t in str(v).lower().replace("_", "-").replace("-", " ").split() if t][:cap]


def render(rec):
    """Fact -> (text, [(word, role)], (entity, slot, value)) or None."""
    e = rec.get("entity") or {}; r = rec.get("relational") or {}
    name = e.get("name")
    if not name:
        return None
    nw = words(name)
    pairs = [(w, "entity.name") for w in nw]
    if r.get("predicate") and r.get("object"):
        pw, ow = words(r["predicate"]), words(r["object"])
        pairs += [(w, "relational.predicate") for w in pw] + [(w, "relational.object") for w in ow]
        slot, val = "relational.object", " ".join(ow)
    elif e.get("kind"):
        kw = words(e["kind"])
        pairs += [("is", "O"), ("a", "O")] + [(w, "entity.kind") for w in kw]
        slot, val = "entity.kind", " ".join(kw)
    else:
        return None
    text = " ".join(w for w, _ in pairs)
    return text, pairs, (" ".join(nw), slot, val)


def main(n, seed=0):
    ensure()
    os.makedirs("data", exist_ok=True)
    rng = random.Random(seed)
    facts, qa = [], []
    with open(SRC) as f:
        for line in f:
            if len(facts) >= n:
                break
            try:
                out = render(json.loads(line))
            except Exception:
                continue
            if not out:
                continue
            text, pairs, (ent, slot, val) = out
            facts.append({"text": text, "word_roles": pairs})
            if slot in QUESTION and val:
                qa.append({"question": QUESTION[slot].format(e=ent),
                           "answer": val, "entity": ent, "slot": slot})
    rng.shuffle(facts); rng.shuffle(qa)
    with open("data/facts.jsonl", "w") as fo:
        for r in facts: fo.write(json.dumps(r) + "\n")
    with open("data/qa.jsonl", "w") as fo:
        for r in qa: fo.write(json.dumps(r) + "\n")

    # report
    from collections import Counter
    role_counts = Counter(role for r in facts for _, role in r["word_roles"])
    print(f"wrote data/facts.jsonl ({len(facts)} facts), data/qa.jsonl ({len(qa)} questions)")
    print("role distribution:", dict(role_counts))
    print("sample fact:", facts[0])
    print("sample qa:", qa[0] if qa else "—")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    main(ap.parse_args().n)
