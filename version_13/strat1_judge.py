"""
Strat 1 — THE JUDGE: Haiku reads every vocab word into the lattice
==================================================================
For each of the ~5.4k vocabulary words, claude-haiku-4-5 fills the fixed
13-role lattice: one NSM-prime value + one centrality weight per role,
returned as structured output (schema-enforced, then re-validated against
strat1_schema). Results append to readings.jsonl — resumable, so a killed
run continues where it left off.

Run:  python strat1_judge.py --pilot 12    # eyeball a sample first
      python strat1_judge.py               # full vocab (~5.4k words)
Out:  readings.jsonl   {"word": ..., "reading": {"<layer.seg.role>":
                        {"value": prime, "weight": 0..1}, ...}}
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, Field, create_model

from step1_data import words as VOCAB
from strat1_schema import LATTICE, ROLE_PATHS, validate_reading

MODEL = "claude-haiku-4-5"
OUT_PATH = Path(__file__).parent / "readings.jsonl"
META_PATH = Path(__file__).parent / "readings_meta.json"
WORKERS = 8

def vocab_fingerprint() -> str:
    """sha256 over the ordered vocab — consumers verify corpus match on load."""
    import hashlib
    return hashlib.sha256("\n".join(VOCAB).encode()).hexdigest()[:16]

# ---- Pydantic output model, generated from the schema ------------------------
def _role_model(layer: str, seg: str, role: str):
    allowed = tuple(LATTICE[layer]["segments"][seg]["roles"][role])
    return create_model(
        f"Role_{layer}_{seg}_{role}",
        value=(Literal[allowed], ...),
        weight=(float, Field(ge=0.0, le=1.0)),
    )

Reading = create_model(
    "Reading",
    **{f"{l}_{s}_{r}": (_role_model(l, s, r), ...) for l, s, r in ROLE_PATHS},
)

FIELD_TO_PATH = {f"{l}_{s}_{r}": f"{l}.{s}.{r}" for l, s, r in ROLE_PATHS}

# ---- Prompt ------------------------------------------------------------------
def _schema_lines() -> str:
    lines = []
    for l, s, r in ROLE_PATHS:
        vals = LATTICE[l]["segments"][s]["roles"][r]
        lines.append(f"- {l}_{s}_{r}: one of {vals}")
    return "\n".join(lines)

SYSTEM = f"""You judge English words into a fixed semantic lattice grounded in the
65 NSM (Natural Semantic Metalanguage) primes. For the given word, fill EVERY
role below with the single best-fitting prime value and a weight.

Roles and allowed values (value MUST come from the role's list):
{_schema_lines()}

Rules:
- weight = how central this role is to the word's core meaning: 1.0 = defining,
  0.5 = clearly present, 0.2 = weakly present, 0.0 = absent.
- Use value "none" (with weight 0.0) when a role does not apply.
- Judge the word's DOMINANT everyday sense in children's stories.
- A concrete noun like "cat": category=something, animacy=live, most other
  roles none. A verb like "shout": agency=do, speech=say. A word like "huge":
  size=big, intensity=very.
- Function words get their logical/temporal/spatial primes: "not" ->
  logical=not (weight 1.0); "yesterday" -> temporal=before.
- Motion words ("run", "jump", "fly"): agency=move. The spatial role is ONLY
  for location/direction primes from its own list — never "move".
- intensity only takes "very" or "more"; if you want another value there,
  it belongs in a different role.
- If the value you instinctively want for a role is NOT in that role's list,
  put "none" (0.0) there — a prime may only appear in roles that list it
  (e.g. "like" belongs to logical, never spatial).
"""

# ---- API ---------------------------------------------------------------------
def _api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", "ANTHROPIC_API_KEY", "-w"],
            capture_output=True, text=True,
        )
        key = out.stdout.strip()
    if not key:
        sys.exit("no ANTHROPIC_API_KEY in env or keychain")
    return key

def judge_word(client: anthropic.Anthropic, word: str) -> dict:
    """One word -> validated reading dict keyed by dotted role path."""
    last_err = None
    for _ in range(3):  # schema is enforced; retries cover transient API errors
        try:
            resp = client.messages.parse(
                model=MODEL,
                max_tokens=2000,
                system=[{"type": "text", "text": SYSTEM,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": f"Word: {word}"}],
                output_format=Reading,
            )
            parsed = resp.parsed_output
            reading = {FIELD_TO_PATH[f]: {"value": getattr(parsed, f).value,
                                          "weight": round(getattr(parsed, f).weight, 2)}
                       for f in FIELD_TO_PATH}
            problems = validate_reading(reading)
            if problems:
                last_err = f"validation: {problems}"
                continue
            return reading
        except Exception as e:  # API errors AND schema/validation misses retry
            last_err = str(e)
    raise RuntimeError(f"{word}: {last_err}")

# ---- Driver ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0,
                    help="judge only N sample words and print them")
    args = ap.parse_args()

    done = set()
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            done = {json.loads(line)["word"] for line in f if line.strip()}

    todo = [w for w in VOCAB if w not in done and w != "<unk>"]
    if args.pilot:
        # spread across the alphabet + include known-interesting words
        seeds = [w for w in ("cat", "run", "happy", "huge", "not", "yesterday",
                             "say", "die", "inside", "maybe", "friend", "shout")
                 if w in todo]
        todo = (seeds + [w for w in todo if w not in seeds])[: args.pilot]
    print(f"vocab={len(VOCAB)}  done={len(done)}  judging={len(todo)}")

    # Corpus-binding metadata: the readings are only valid for THIS vocab.
    META_PATH.write_text(json.dumps({
        "model": MODEL, "vocab_sha": vocab_fingerprint(),
        "vocab_size": len(VOCAB), "corpus": "tinystories_valid",
    }, indent=2))

    client = anthropic.Anthropic(api_key=_api_key())
    ok = err = 0
    with OUT_PATH.open("a") as out, ThreadPoolExecutor(WORKERS) as pool:
        futs = {pool.submit(judge_word, client, w): w for w in todo}
        for fut in as_completed(futs):
            word = futs[fut]
            try:
                reading = fut.result()
            except Exception as e:
                err += 1
                print(f"  FAIL {word}: {e}", file=sys.stderr)
                continue
            out.write(json.dumps({"word": word, "reading": reading}) + "\n")
            out.flush()
            ok += 1
            if args.pilot:
                active = {k.split(".")[-1]: f"{v['value']}:{v['weight']}"
                          for k, v in reading.items() if v["value"] != "none"}
                print(f"  {word:<12} {active}")
            elif ok % 100 == 0:
                print(f"  {ok}/{len(todo)}")
    print(f"done: {ok} ok, {err} failed -> {OUT_PATH.name}")

if __name__ == "__main__":
    main()
