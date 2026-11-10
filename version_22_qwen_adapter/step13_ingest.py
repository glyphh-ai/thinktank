"""step13_ingest.py — ingest: the model rewrites free text into facts the memory can hold.

step12: notes taught whole are recalled 6-28% when they say "they" after
the first sentence, against 99-100% for single sentences that name their
subject. So before anything is told to the memory, the model rewrites each
note into such sentences, and plain code checks every line before it is
kept. This measures both halves on step12's "far" notes:

  rewrite    one note per call, a general instruction (nothing about
             managers, towns, products or codes): one short sentence per
             line, each naming in full who or what it is about, only what
             the note says
  check      a line is kept only if every word in it that is not ordinary
             English (a name, a value) appears in the note, and it does not
             open with a pronoun. No model in the check.
  counted    of the four facts in each note: stated in a kept line with the
             customer's name (kept), missing (dropped); lines with a word
             the note never had (invented, before the check); a value put
             with another fact's wording (mis-bound)

Then the memory (step4c's: 8-word window, delta 0.05) is taught three ways
and asked step12's eight untaught questions:
  whole      the notes as they are (step12's far/window)
  rewritten  the kept lines, each told on its own
  clean      the four facts as hand-made one-liners: the ceiling

Corrections: a third of the cases get step12's update note ("An update came
in for {p}. Their account manager is now {v}."), rewritten and checked the
same way, told, and the manager asked again.

Run:  QWEN=Qwen3-14B-4bit python step13_ingest.py
"""

import argparse
import json
import os
import re
import time

import numpy as np

import qwen
import step3_glyphh as s3
from step1_adapter import Arm, HERE, book, recall
from step5_instructions import chat, generate
from step12_unstructured import ASK, ATTRS, NEAR, UPDATE, groups, make_cases, note, tau_for

ASKED = ("Rewrite the note below as a list of facts. One short sentence per line. Every sentence must name in "
         "full the person or thing it is about; never write 'they', 'she', 'he', 'it' or 'the customer'. Use "
         "only what the note says, and leave out anything that is not a lasting fact.\n\nNote: {note}")
# v2 (after v1 left half the facts without their subject: the model fixed "They" but copied "Home is the town
# of Huraku." as it stood, and wrote updates value-first, "Bada is now the account manager for ..."): the model
# names the subject once, every fact must START with it, and the check enforces that. Still nothing about
# managers, towns, products or codes.
ASKED2 = ("Read the note below. On the first line write 'Subject:' and the full name of the person or thing the "
          "note is about. Then write its lasting facts, one short sentence per line. Every sentence must START "
          "with the subject's full name, for example \"<name>'s <something> is <value>.\" or \"<name> <verb> "
          "<value>.\" Use only what the note says, and leave out passing events.\n\nNote: {note}")
PRONOUNS = {"they", "their", "she", "he", "her", "his", "it", "its", "them"}
CUES = {"manager": {"manager", "manages", "managed"},
        "town": {"town", "lives", "live", "living", "home", "resides", "city", "based", "located"},
        "product": {"product", "bought", "owns", "own", "purchased", "item"},
        "code": {"code"}}


def unknown(w):
    return s3.LEXICON.entry(w) is None and s3.LEXICON.lemma(w) is None


V2 = False


def rewrite(q, text):
    out, _ = generate(q, chat(q, [{"role": "user", "content": (ASKED2 if V2 else ASKED).format(note=text)}]),
                      max_new=200)
    lines = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", x).strip() for x in out.splitlines()]
    lines = [x for x in lines if x]
    if V2:
        # the check's second half: the subject the model named must be in the note, and only lines that start
        # with it are kept
        subject = lines[0].split(":", 1)[1].strip().rstrip(".") if lines and lines[0].lower().startswith("subject:") else ""
        ok = len(subject) > 2 and subject in text
        return [x for x in lines[1:] if ok and x.startswith(subject)]
    return lines


def check(line, text):
    """keep a line only if its names and values are the note's own, and it names its subject."""
    words = s3._WORD.findall(line.lower())
    have = set(s3._WORD.findall(text.lower()))
    if len(words) < 3 or words[0] in PRONOUNS:
        return False
    return all(w in have for w in words if unknown(w))


def ingest(q, cases, texts):
    """rewrite and check every note; the kept lines and what happened to each fact."""
    kept, stats = [], {"lines": 0, "kept_lines": 0, "invented": 0, "kept": 0, "dropped": 0, "misbound": 0}
    for c, text in zip(cases, texts):
        lines = rewrite(q, text)
        have = set(s3._WORD.findall(text.lower()))
        stats["lines"] += len(lines)
        stats["invented"] += sum(any(unknown(w) and w not in have for w in s3._WORD.findall(x.lower())) for x in lines)
        good = [x for x in lines if check(x, text)]
        stats["kept_lines"] += len(good)
        for a in ATTRS:
            hit = [x for x in good if c["p"] in x and c[a] in x]
            stats["kept" if hit else "dropped"] += 1
            for x in good:
                w = set(s3._WORD.findall(x.lower()))
                if c[a] in x and not (w & CUES[a]) and any(w & CUES[b] for b in ATTRS if b != a):
                    stats["misbound"] += 1
        kept.append(good)
    return kept, stats


def ask_all(arm, cases, pick):
    r = np.random.default_rng(17)
    row = {f"{a}_{tag}": recall(arm.answer_bits(groups(cases, pick, a, form, r)))[0]
           for a in ATTRS for form, tag in enumerate(("qa", "reworded"))}
    row["recall"] = float(np.mean(list(row.values())))
    return row


def main(n, sample, out, delta=0.05, only=None):
    q = qwen.Qwen()
    cases = make_cases(n)
    rng = np.random.default_rng(11)
    texts = [note(c, "far", rng) for c in cases]
    pick = np.random.default_rng(3).choice(n, min(sample, n), replace=False)
    print(f"{qwen.NAME}; {n} notes ({4 * n} facts), {sample} cases asked, e.g.\n  note: {texts[0]}", flush=True)

    t = time.time()
    kept, st = ingest(q, cases, texts)
    secs = time.time() - t
    print("  rewritten, kept: " + " | ".join(kept[0]))
    print(f"\ningest: {secs / n:.1f}s a note; {st['lines']} lines written, {st['kept_lines']} kept by the check "
          f"({st['invented']} had a word the note never did); of {4 * n} facts {st['kept']} kept "
          f"({st['kept'] / (4 * n):.1%}), {st['dropped']} dropped, {st['misbound']} mis-bound\n", flush=True)
    results = {"notes": n, "ingest": st, "ingest_s_per_note": secs / n, "example": {"note": texts[0], "kept": kept[0]}}

    tau_whole = tau_for(q, s3.GlyphhMemory, n, "far", delta)
    dev = s3.GlyphhMemory(q.tok, delta=delta)
    for c in make_cases(n, seed=1):
        for a in ATTRS:
            dev.learn(q.encode(NEAR[a][0].format(p=c["p"], v=c[a])))
    _, s = dev.nearest(dev.contexts(q.encode(book("frankenstein", 84)[50000:58000])))
    tau_lines = float(np.percentile(s, 99.5))
    del dev

    third = n // 3
    fixed = np.random.default_rng(4).choice(third, min(sample, third), replace=False)
    rest = third + np.random.default_rng(5).choice(n - third, min(sample, n - third), replace=False)
    updates = [UPDATE["far"].format(p=c["p"], v=c["new"]) for c in cases[:third]]
    t = time.time()
    up_kept, up_st = ingest(q, [{**c, "manager": c["new"]} for c in cases[:third]], updates)
    print(f"updates: {third} rewritten in {time.time() - t:.0f}s, e.g. {updates[0]!r} -> {up_kept[0]}; "
          f"manager kept in {sum(any(c['p'] in x and c['new'] in x for x in k) for c, k in zip(cases, up_kept))}"
          f" of {third}\n", flush=True)
    results["updates"] = {"example": [updates[0], up_kept[0]], **up_st}

    arms = [("whole", tau_whole, texts, updates),
            ("rewritten", tau_lines, [x for k in kept for x in k], [x for k in up_kept for x in k]),
            ("clean", tau_lines, [NEAR[a][0].format(p=c["p"], v=c[a]) for c in cases for a in ATTRS],
             [NEAR["manager"][0].format(p=c["p"], v=c["new"]) for c in cases[:third]])]
    for name, tau, teach, fix in arms:
        if only and name not in only:
            continue
        t = time.time()
        mem = s3.GlyphhMemory(q.tok, delta=delta)
        mem.tau = tau
        arm = Arm(q, name, exact=mem)
        arm.teach(teach)
        row = {"arm": name, "tau": tau, "keys": len(mem.counts), **ask_all(arm, cases, pick)}
        arm.teach(fix)
        r2 = np.random.default_rng(23)
        for form, tag in enumerate(("qa", "reworded")):
            nw, old = recall(arm.answer_bits(groups(cases, fixed, "manager", form, r2, "new", "manager")), True)
            row[f"new_{tag}"], row[f"old_{tag}"] = nw, old
            row[f"untouched_{tag}"], _ = recall(arm.answer_bits(groups(cases, rest, "manager", form, r2)))
        print(f"{name:<10} recall {row['recall']:4.0%}  ("
              + " ".join(f"{a} {row[a + '_qa']:.0%}/{row[a + '_reworded']:.0%}" for a in ATTRS)
              + f")  tau {tau:.2f}, {row['keys']:,} keys | corrected: new {row['new_qa']:.0%}/{row['new_reworded']:.0%}"
              f" old {row['old_qa']:.0%}/{row['old_reworded']:.0%} untouched {row['untouched_qa']:.0%}/"
              f"{row['untouched_reworded']:.0%}  ({time.time() - t:.0f}s)", flush=True)
        results[name] = row
        with open(os.path.join(HERE, out), "w") as fh:
            json.dump(results, fh, indent=1, default=float)
        del mem, arm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", type=int, default=300)
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--out", default="step13_results.json")
    ap.add_argument("--v2", action="store_true", help="the subject-first rewrite and check")
    ap.add_argument("--arms", default="", help="e.g. rewritten (default: all three)")
    a = ap.parse_args()
    V2 = a.v2
    main(a.notes, a.sample, a.out, only=a.arms.split(",") if a.arms else None)
