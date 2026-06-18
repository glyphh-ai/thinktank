"""
enrich.py — gold labels for the messy test, via Ada's REAL enricher.

For each messy summary, call adaL's UniversalEnricher (LLM-at-write-time) to map
the prose -> Ada's 7-layer/33-role schema. That enricher output IS the gold (it
is, by definition, what "correct extraction" means for Ada). -> gold.jsonl
{title, text, slots}.

REQUIRES: ANTHROPIC_API_KEY (the enricher calls the API), and the adaL repo
(auto-cloned). This is the messy task's labeling step — can't run without a key.

Run:  ANTHROPIC_API_KEY=... python enrich.py --in messy.jsonl --out gold.jsonl
"""

import os, sys, json, argparse, subprocess

ADAL = "/tmp/adaL"


def ensure_adal():
    if not os.path.exists(ADAL):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/glyphh-ai/adaL", ADAL],
                       check=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    sys.path.insert(0, ADAL)


def main(inp, out):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set — the enricher needs it to label gold.")
    ensure_adal()
    from ada.cognitive.universal import UniversalEnricher   # Ada's REAL enricher
    enr = UniversalEnricher()                                # model from ADA_MODEL/default
    n = 0
    with open(inp) as fi, open(out, "w") as fo:
        for line in fi:
            rec = json.loads(line)
            try:
                slots = enr.enrich(rec["text"])              # prose -> {layer:{role:value}}
            except Exception as e:
                print("skip", rec["title"], e); continue
            fo.write(json.dumps({"title": rec["title"], "text": rec["text"],
                                 "slots": slots}) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"enriched {n}...")
    print(f"wrote {n} gold (text, slots) pairs -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="messy.jsonl")
    ap.add_argument("--out", default="gold.jsonl")
    a = ap.parse_args()
    main(a.inp, a.out)
