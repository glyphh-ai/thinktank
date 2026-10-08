"""
ADA-HAIKU — build the big word base: a v3 glyph for (nearly) every English word
==============================================================================
Seeds: every single-word WordNet lemma (lowercase letters only) + the
TinyStories vocabularies + the existing glyph words. Then a closure round for
value words that still have no glyph.

  model    claude-haiku-5-5, thinking off, structured output (27 strings),
           the glyph_schema3 prompt, cached; Message Batches API (50% off)
  budget   HARD CAP (default $20): the first batch's measured cost per glyph
           projects the rest; nothing is submitted that would cross the cap
  capture  nothing is lost —
             · every result is appended to raw/<batch>.jsonl as it downloads
             · each finished batch becomes shards/<batch>.jsonl.gz, committed
               and pushed with state.json
             · state.json (committed) holds every batch id: results stay on the
               API for 29 days, so a dead container resumes and re-downloads
  watch    PROGRESS.live.md (rewritten every poll, local) and PROGRESS.md
           (snapshot committed + pushed on every submit and download)

Run (background):  nohup python ada_haiku/build.py > ada_haiku/log.txt 2>&1 &
Resume:            same command — it reads state.json and carries on
Env:               ANTHROPIC_API_KEY
"""

import argparse
import gzip
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import anthropic                                                # noqa: E402
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming  # noqa: E402
from anthropic.types.messages.batch_create_params import Request  # noqa: E402

import closure as C                                             # noqa: E402
import glyph_schema3 as S                                       # noqa: E402

MODEL = "claude-haiku-5-5"
# Batch API = 50% of standard: input $0.10, output $0.50, cache write 1.25x, read 0.1x (per MTok)
PRICE = {"in": 0.05, "cw": 0.0625, "cr": 0.005, "out": 0.25}
STATE = HERE / "state.json"                 # committed snapshot (written at push time)
STATE_LIVE = HERE / "state.local.json"      # the working copy, saved every poll (untracked)
PROGRESS = HERE / "PROGRESS.md"             # committed (written at push time)
PROGRESS_LIVE = HERE / "PROGRESS.live.md"   # rewritten every poll (untracked)
RAW, SHARDS = HERE / "raw", HERE / "shards"
WORD_RE = re.compile(r"^[a-z]+$")


# ============================================================================
# seeds and state
# ============================================================================
def seed_words():
    from nltk.corpus import wordnet as wn
    words = set()
    for s in wn.all_synsets():
        for l in s.lemma_names():
            if WORD_RE.match(l):
                words.add(l)
    for p in (ROOT.parent / "version_16_base/data/cache_f20/vocab.json",):
        if p.exists():
            words |= {w for w in json.loads(p.read_text()) if WORD_RE.match(w)}
    for line in open(ROOT / "readings.jsonl"):                  # v13 TinyStories vocab
        w = json.loads(line)["word"]
        if WORD_RE.match(w):
            words.add(w)
    for line in open(ROOT / "glyphs3.jsonl"):
        words.add(json.loads(line)["word"])
    return sorted(words)


def load_state():
    for p in (STATE_LIVE, STATE):                # live copy first; committed one on a new container
        if p.exists():
            return json.loads(p.read_text())
    return {"created": now(), "model": MODEL, "cap_usd": None, "batches": [],
            "cost_usd": 0.0, "glyphs": 0, "failed": [], "rounds": {}}


def save_state(st):
    STATE_LIVE.write_text(json.dumps(st, indent=1))


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def done_words():
    out = set()
    for f in sorted(RAW.glob("*.jsonl")):
        for line in open(f):
            try:
                out.add(json.loads(line)["word"])
            except json.JSONDecodeError:
                pass                                            # a torn last line on crash
    return out


# ============================================================================
# requests
# ============================================================================
WRITER = C.Writer("glyph_schema3")
FIELDS = list(WRITER.fields)
SCHEMA = {"type": "object", "properties": {f: {"type": "string"} for f in FIELDS},
          "required": FIELDS, "additionalProperties": False}
SYSTEM = [{"type": "text", "text": WRITER.system, "cache_control": {"type": "ephemeral"}}]


def request(cid, word):
    return Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
        model=MODEL, max_tokens=1500, thinking={"type": "disabled"}, system=SYSTEM,
        messages=[{"role": "user", "content": f"Word: {word}"}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}}))


def to_glyph(word, data):
    g = {WRITER.fields[f]: {"value": str(data[f]).strip().lower(), "weight": 1.0} for f in FIELDS}
    return g, S.validate_glyph(word, g)


def cost(u):
    return (u.input_tokens * PRICE["in"] + (u.cache_creation_input_tokens or 0) * PRICE["cw"]
            + (u.cache_read_input_tokens or 0) * PRICE["cr"] + u.output_tokens * PRICE["out"]) / 1e6


# ============================================================================
# progress + capture
# ============================================================================
def write_progress(st, total_seeds, note=""):
    open_b = [b for b in st["batches"] if b["status"] != "downloaded"]
    per = st["cost_usd"] / max(st["glyphs"], 1)
    lines = [
        "# ada-haiku build — progress", "",
        f"_updated {now()}_ — model `{MODEL}`, Batch API, hard cap **${st['cap_usd']:.2f}**", "",
        "| | |", "|---|---|",
        f"| glyphs captured | **{st['glyphs']:,}** of {total_seeds:,} seeds |",
        f"| spent so far | **${st['cost_usd']:.2f}** (${per:.5f}/glyph) |",
        f"| failed validation (queued for retry) | {len(st['failed']):,} |",
        f"| batches | {len(st['batches'])} submitted, {len(open_b)} still open |", "",
        "| batch | round | words | status | succeeded | errored | submitted |",
        "|---|---|---|---|---|---|---|"]
    for b in st["batches"]:
        lines.append(f"| `{b['id'][-12:]}` | {b['round']} | {b['n']:,} | {b['status']} | "
                     f"{b.get('succeeded', '')} | {b.get('errored', '')} | {b['submitted']} |")
    if note:
        lines += ["", f"**{note}**"]
    lines += ["", "Results: `shards/*.jsonl.gz` (one per finished batch). "
              "Batch ids above are re-downloadable from the API for 29 days."]
    PROGRESS_LIVE.write_text("\n".join(lines) + "\n")


def git_push(msg):
    """Commit the capture (shards, request maps, state, progress) and push.
    Never commits raw/ or the key."""
    try:
        STATE.write_text(STATE_LIVE.read_text())                 # snapshot the live files
        PROGRESS.write_text(PROGRESS_LIVE.read_text())
        paths = [p for p in (HERE / "shards", HERE / "requests", STATE, PROGRESS) if p.exists()]
        subprocess.run(["git", "-C", str(ROOT.parent), "add", *map(str, paths)],
                       check=True, capture_output=True)
        r = subprocess.run(["git", "-C", str(ROOT.parent), "commit", "-q", "-m",
                            f"ada-haiku: {msg}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
                            "Claude-Session: https://claude.ai/code/session_01VUsxdcdhXYHqLqZHQQHBh4"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            for delay in (0, 2, 4, 8, 16):
                time.sleep(delay)
                p = subprocess.run(["git", "-C", str(ROOT.parent), "push", "-q", "origin", "HEAD"],
                                   capture_output=True, text=True)
                if p.returncode == 0:
                    break
            print(f"[{now()}] pushed: {msg}", flush=True)
    except Exception as e:                                       # capture is on disk regardless
        print(f"[{now()}] git push failed (data safe on disk): {e}", flush=True)


# ============================================================================
# the loop
# ============================================================================
def download(client, st, b):
    raw = RAW / f"{b['id']}.jsonl"
    have = set()
    if raw.exists():
        have = {json.loads(l)["cid"] for l in open(raw) if l.strip()}
    words = json.loads((HERE / "requests" / f"{b['round']}_{b['part']}.json").read_text())
    ok = bad = 0
    with open(raw, "a") as out:
        for res in client.messages.batches.results(b["id"]):
            if res.custom_id in have:
                continue
            word = words[res.custom_id]
            if res.result.type != "succeeded":
                st["failed"].append(word); bad += 1
                continue
            msg = res.result.message
            st["cost_usd"] += cost(msg.usage)
            try:
                data = json.loads(next(x.text for x in msg.content if x.type == "text"))
                glyph, problems = to_glyph(word, data)
            except Exception as e:
                glyph, problems = None, [str(e)]
            if problems:
                st["failed"].append(word); bad += 1
                continue
            out.write(json.dumps({"cid": res.custom_id, "word": word, "glyph": glyph}) + "\n")
            out.flush()
            ok += 1
    SHARDS.mkdir(exist_ok=True)
    with open(raw) as f, gzip.open(SHARDS / f"{b['id']}.jsonl.gz", "wt") as g:
        for line in f:
            d = json.loads(line)
            g.write(json.dumps({"word": d["word"], "glyph": d["glyph"]}) + "\n")
    b["status"], b["downloaded"] = "downloaded", now()
    st["glyphs"] = len(done_words())
    return ok, bad


def submit(client, st, rnd, part, words):
    (HERE / "requests").mkdir(exist_ok=True)
    cids = {f"{rnd}-{part}-{i}": w for i, w in enumerate(words)}
    (HERE / "requests" / f"{rnd}_{part}.json").write_text(json.dumps(cids))
    batch = client.messages.batches.create(requests=[request(c, w) for c, w in cids.items()])
    st["batches"].append({"id": batch.id, "round": rnd, "part": part, "n": len(words),
                          "status": batch.processing_status, "submitted": now()})
    save_state(st)
    print(f"[{now()}] submitted {batch.id}: round {rnd} part {part}, {len(words)} words", flush=True)
    write_progress(st, sum(b["n"] for b in st["batches"]))
    git_push(f"submitted {batch.id[-12:]} ({rnd} part {part}, {len(words)} words)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=20.0)
    ap.add_argument("--batch", type=int, default=8000)
    ap.add_argument("--pilot", type=int, default=2000, help="first batch size (measures cost)")
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()

    RAW.mkdir(exist_ok=True)
    client = anthropic.Anthropic()
    st = load_state()
    st["cap_usd"] = args.cap
    seeds = seed_words()
    total = len(seeds)
    print(f"[{now()}] seeds: {total:,} words; cap ${args.cap:.2f}", flush=True)

    rnd = "r1"
    while True:
        # 1. poll open batches; download finished ones
        for b in st["batches"]:
            if b["status"] == "downloaded":
                continue
            info = client.messages.batches.retrieve(b["id"])
            b["status"] = info.processing_status
            b["succeeded"], b["errored"] = info.request_counts.succeeded, info.request_counts.errored
            if info.processing_status == "ended":
                ok, bad = download(client, st, b)
                save_state(st)
                write_progress(st, total)
                git_push(f"batch {b['id'][-12:]} ({b['round']}): +{ok} glyphs, {bad} failed, "
                         f"${st['cost_usd']:.2f} spent")
        save_state(st)

        # 2. decide what to submit next
        done = done_words()
        pending_ids = {w for b in st["batches"] if b["status"] != "downloaded"
                       for w in json.loads((HERE / "requests" / f"{b['round']}_{b['part']}.json")
                                           .read_text()).values()}
        todo = [w for w in seeds if w not in done and w not in pending_ids]
        open_n = sum(b["n"] for b in st["batches"] if b["status"] != "downloaded")
        if not todo and open_n == 0 and rnd == "r1":             # closure round: value words
            vals = set()
            for f in sorted(SHARDS.glob("*.jsonl.gz")):
                for line in gzip.open(f, "rt"):
                    vals |= {e["value"] for e in json.loads(line)["glyph"].values()}
            todo = sorted(vals - done - pending_ids)
            if todo:
                rnd = "r2"
                seeds = seeds + todo
                total = len(seeds)
        retry = sorted(set(st["failed"]) - done - pending_ids)
        if not todo and open_n == 0 and retry and not st["rounds"].get("retried"):
            todo, st["rounds"]["retried"] = retry, True
            st["failed"] = []
            rnd = rnd + "-retry"

        per = st["cost_usd"] / st["glyphs"] if st["glyphs"] else None
        note = ""
        if todo:
            if not st["batches"]:                                # pilot: measure cost first
                submit(client, st, rnd, 0, todo[:args.pilot])
            elif per is None:
                note = "waiting for the pilot batch to measure cost per glyph"
            else:
                projected = st["cost_usd"] + (open_n + min(len(todo), args.batch)) * per * 1.1
                if projected > args.cap:
                    note = (f"STOPPED AT CAP: next batch would project ${projected:.2f} > "
                            f"${args.cap:.2f}; {len(todo):,} words left unglyphed")
                elif open_n < 3 * args.batch:                    # keep a few batches in flight
                    part = len([b for b in st["batches"] if b["round"] == rnd])
                    submit(client, st, rnd, part, todo[:args.batch])
        write_progress(st, total, note)
        if not todo and open_n == 0:
            write_progress(st, total, "DONE — every seed and closure word is captured")
            git_push(f"DONE: {st['glyphs']:,} glyphs, ${st['cost_usd']:.2f}")
            break
        if note.startswith("STOPPED AT CAP") and open_n == 0:
            git_push(f"stopped at cap: {st['glyphs']:,} glyphs, ${st['cost_usd']:.2f}")
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
