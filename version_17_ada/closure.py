"""
THE CLOSURE — an LLM writes every word's glyph, down the chain, until closed
===========================================================================
Round 1 glyphs the seed words. Every value word used in those glyphs that
has no glyph yet is glyphed in round 2, and so on until a round adds no new
words: then every word used to describe a word has its own glyph.

claude-haiku-4-5 writes each glyph as structured output (schema-enforced,
then re-validated by the schema's validate_glyph). Results append to
glyphs.jsonl — resumable, so a killed run continues where it left off.

Run:  python closure.py --words love hate kind mean --rounds 1   # pilot
      python closure.py                                          # vocab -> closed
      python closure.py --schema glyph_schema2 --out glyphs2.jsonl \
             --with-schema-words --words love hate kind mean     # v2 schema
Out:  glyphs.jsonl  {"word": ..., "round": n, "glyph": {"<layer.seg.role>":
                     {"value": word|"none", "weight": 0..1}, ...}}
Env:  ANTHROPIC_API_KEY
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import importlib

import anthropic
from pydantic import BaseModel, Field, create_model

MODEL = "claude-haiku-4-5"
HERE = Path(__file__).parent
WORKERS = 8


class RoleFill(BaseModel):
    value: str
    weight: float = Field(ge=0.0, le=1.0)


class RoleValue(BaseModel):          # unweighted schemas: every role weighs the same
    value: str


class Writer:
    """Everything that depends on the schema: the output model and the prompt."""
    def __init__(self, schema_module: str):
        S = importlib.import_module(schema_module)
        self.S = S
        self.fields = {f"{l}_{s}_{r}": f"{l}.{s}.{r}" for l, s, r in S.ROLE_PATHS}
        self.weighted = getattr(S, "WEIGHTED", True)
        fill = RoleFill if self.weighted else RoleValue
        self.model = create_model("Glyph", **{f: (fill, ...) for f in self.fields})
        lines = "\n".join(f"- {l}_{s}_{r}: {S.SCHEMA[l][s][r]}" for l, s, r in S.ROLE_PATHS)
        intro = getattr(S, "PROMPT_INTRO", """You write the glyph of an English word: its meaning as role: value
pairs in a fixed schema that is the same for every word. Fill EVERY role.""")
        examples = getattr(S, "PROMPT_EXAMPLES", """- Function words get glyphs too: "not" -> category: word, opposite: yes,
  thought: denial; "under" -> category: position, opposite: over, place: below.""")
        weight_rule = ("""- weight = how central this role is to the word: 1.0 = defining,
  0.5 = clearly present, 0.2 = weakly present. Use "none" with weight 0.0
  when a role does not apply. Most words fill well under half the roles.
""" if self.weighted else "")
        self_rule = "- A word may never be its own value.\n" if self.weighted else ""
        none_rule = ' or "none"' if self.weighted else ""
        self.system = f"""{intro}

Roles:
{lines}

Rules:
- value is ONE lowercase English word (letters only, no spaces){none_rule}.
- Prefer simple, common everyday words as values — the words a child knows.
  Use base forms: singular nouns, plain verbs ("run", not "running").
{self_rule}{weight_rule}- Describe the word's DOMINANT everyday sense (as in children's stories).
{examples}
"""

    def write(self, client: anthropic.Anthropic, word: str) -> dict:
        """One word -> validated glyph keyed by dotted role path."""
        last_err = None
        for _ in range(3):
            try:
                resp = client.messages.parse(
                    model=MODEL,
                    max_tokens=2500,
                    system=[{"type": "text", "text": self.system,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": f"Word: {word}"}],
                    output_format=self.model,
                )
                g = resp.parsed_output
                glyph = {}
                for f, path in self.fields.items():
                    v = getattr(g, f).value.strip().lower()
                    w = round(getattr(g, f).weight, 2) if self.weighted else 1.0
                    glyph[path] = {"value": v, "weight": 0.0 if v == "none" else w}
                problems = self.S.validate_glyph(word, glyph)
                if problems:
                    last_err = f"validation: {problems}"
                    continue
                return glyph
            except Exception as e:  # API errors and schema misses retry
                last_err = str(e)
        raise RuntimeError(f"{word}: {last_err}")


def load_glyphs(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return {d["word"]: d for d in map(json.loads, f) if d}


def value_words(glyphs: dict) -> set:
    return {e["value"] for d in glyphs.values() for e in d["glyph"].values()
            if e["value"] != "none"}


def seed_words() -> list:
    from step1_data import words as VOCAB
    return [w for w in VOCAB if w.replace("'", "").isalpha()]   # drop punctuation, <eot>, <unk>


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", nargs="*", help="seed words (default: the whole vocab)")
    ap.add_argument("--rounds", type=int, default=50, help="stop after this many rounds")
    ap.add_argument("--max-glyphs", type=int, default=0,
                    help="stop before a round that would pass this many glyphs (cost cap)")
    ap.add_argument("--schema", default="glyph_schema", help="schema module (glyph_schema2 = v2)")
    ap.add_argument("--out", default="glyphs.jsonl")
    ap.add_argument("--with-schema-words", action="store_true",
                    help="also seed every layer/segment/role name (schema.SCHEMA_WORDS)")
    args = ap.parse_args()

    writer = Writer(args.schema)
    out_path = HERE / args.out
    client = anthropic.Anthropic()
    glyphs = load_glyphs(out_path)
    seeds = list(args.words or seed_words())
    if args.with_schema_words:
        seeds += [w for w in writer.S.SCHEMA_WORDS if w not in seeds]
    pending = [w for w in seeds if w not in glyphs]
    rnd = max((d["round"] for d in glyphs.values()), default=0)
    if not pending:   # resuming: continue from the open frontier
        pending = sorted(value_words(glyphs) - glyphs.keys())

    failed = set()
    with out_path.open("a") as out:
        while pending and rnd < args.rounds:
            if args.max_glyphs and len(glyphs) + len(pending) > args.max_glyphs:
                print(f"cap: next round would reach {len(glyphs) + len(pending)} glyphs "
                      f"> --max-glyphs {args.max_glyphs}; stopping", flush=True)
                break
            rnd += 1
            print(f"round {rnd}: writing {len(pending)} glyphs "
                  f"(have {len(glyphs)})", flush=True)
            with ThreadPoolExecutor(WORKERS) as pool:
                futs = {pool.submit(writer.write, client, w): w for w in pending}
                for i, fut in enumerate(as_completed(futs), 1):
                    word = futs[fut]
                    try:
                        rec = {"word": word, "round": rnd, "glyph": fut.result()}
                    except Exception as e:
                        failed.add(word)
                        print(f"  FAIL {e}", file=sys.stderr)
                        continue
                    glyphs[word] = rec
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    if i % 250 == 0:
                        print(f"  {i}/{len(pending)}", flush=True)
            pending = sorted(value_words(glyphs) - glyphs.keys() - failed)
            print(f"round {rnd} done: {len(glyphs)} glyphs, "
                  f"{len(pending)} new value words to glyph", flush=True)

    status = "CLOSED" if not pending else f"OPEN ({len(pending)} words unglyphed)"
    print(f"\n{status}: {len(glyphs)} glyphs after {rnd} rounds, {len(failed)} failed")


if __name__ == "__main__":
    main()
