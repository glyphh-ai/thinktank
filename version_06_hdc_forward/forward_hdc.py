"""
forward_hdc.py — HDC in the forward pass: entity-keyed groundedness gate.
=========================================================================
Corrected design (see memory.py): groundedness is NOT whole-sentence cosine. It
is entity-keyed retrieval. As we decode word by word:

  1. the role head names the Ada SLOT of the next word,
  2. the LM head PROPOSES fillers (top-k),
  3. the HDC memory DISPOSES: for the current subject ENTITY, probe(entity, slot)
     = unbind(record[entity], role) -> cleanup -> the stored filler(s). A proposed
     filler is admissible iff it is (near) the cleanup winner for that (entity,slot).

So at each content slot the grounded choices ARE probe(entity, slot); there is no
sentence matching and no exhaustive backtracking. Refusal is geometric: if no
top-k proposal is admissible for (entity, slot), the reachable cell is empty ->
refuse (the doc's empty-cell invariant), rather than assert a fluent falsehood.

The two layers, cleanly separated (as the doc frames it):
  LM head  = fluency / proposals.
  HDC gate = grounded membership / disposal, BEFORE each word commits.

Run:  python forward_hdc.py
"""
from __future__ import annotations

import numpy as np

import model as M
from corpus import build_corpus, ROLES, N_ROLE, SEP
from memory import FactMemory

T = 16
C = 64
STEPS = 2000
SEED = 0
TOPK = 8              # LM proposals the HDC gate re-scores per slot
MARGIN = 0.05        # cleanup winner must beat runner-up by this to be "grounded"
ADMIT = 0.10         # a proposed filler is admissible if its membership >= this


def subject_of(prompt_words, mem, stoi):
    """The entity that keys retrieval = the first prompt word that is a known entity."""
    for w in prompt_words:
        if w in mem.records:
            return w
    return None


def hdc_decode(p, mem, stoi, itos, prompt_words, maxwords=8, verbose=True):
    def wid(w): return stoi.get(w, stoi.get("<unk>", 0))
    prompt_ids = [wid(w) for w in prompt_words]
    entity = subject_of(prompt_words, mem, stoi)

    events = [("prompt", " ".join(prompt_words), entity)]
    words: list[str] = []
    asserted = False

    steps = 0
    while len(words) < maxwords and steps < 40:
        steps += 1
        ctx = prompt_ids + [wid(w) for w in words]
        lm_logits, role_logits = M.next_dual(p, ctx, T)
        slot = ROLES[int(role_logits.argmax())]
        order = [int(cc) for cc in np.argsort(lm_logits)[::-1]]

        if slot == "O":
            w = itos[order[0]]
            if w == SEP:
                events.append(("end", None))
                asserted = True
                break
            words.append(w)
            events.append(("O", w))
            continue

        # content slot: the HDC memory says which fillers are grounded for (entity, slot)
        grounded = mem.probe(entity, slot, topn=3) if entity else []
        gset = {f: s for f, s in grounded}
        lm_cands = [itos[cc] for cc in order[:TOPK] if itos[cc] != SEP]

        # admit an LM proposal only if the entity's record actually returns it for this slot
        pick = None
        for w in lm_cands:
            if entity is None:
                break
            mship = mem.verify(entity, slot, w, margin=MARGIN)[1]
            if w in gset and mship >= ADMIT:
                pick = (w, mship)
                break

        if pick is None:
            # no admissible grounded filler among the proposals -> empty cell -> refuse
            top_probe = grounded[0] if grounded else None
            events.append(("refuse", slot, entity, top_probe,
                           lm_cands[0] if lm_cands else None))
            break

        w, mship = pick
        lm_rank = lm_cands.index(w)
        words.append(w)
        events.append(("word", w, slot, mship, lm_rank,
                       [(f, s) for f, s in grounded[:2]]))

    return prompt_words + words, events, asserted


def print_trace(words, events, asserted):
    for e in events:
        k = e[0]
        if k == "prompt":
            print(f"    prompt '{e[1]}'   subject={e[2]}")
        elif k == "O":
            print(f"    ·     {e[1]:12} (function word)")
        elif k == "word":
            probe = ", ".join(f"{f}({s:+.3f})" for f, s in e[5])
            tag = "" if e[4] == 0 else f"  (HDC vetoed LM top, took rank {e[4]})"
            print(f"    WORD  {e[1]:12} slot={e[2]:20} membership={e[3]:+.3f}  "
                  f"[probe: {probe}]{tag}")
        elif k == "refuse":
            slot, ent, top, lmtop = e[1], e[2], e[3], e[4]
            tp = f"record's best is '{top[0]}'({top[1]:+.3f})" if top else "record is empty"
            print(f"    REFUSE slot={slot} for '{ent}': LM wanted '{lmtop}' but {tp} "
                  f"-> no grounded filler proposed (empty cell)")
        elif k == "end":
            pass
    sent = " ".join(w for w in words if w != SEP)
    print(f"  => {'ASSERTED: ' + sent if asserted else 'REFUSED: ' + sent + ' …'}\n")


if __name__ == "__main__":
    print("building corpus + distilling entity records...")
    c = build_corpus(repeat=60)
    ids, rids, vocab, stoi = c["ids"], c["rids"], c["vocab"], c["stoi"]
    itos = {i: w for w, i in stoi.items()}
    mem = FactMemory(c["enc"]).distill(c["fillers"])
    print(f"  {len(mem.records)} entity records, {len(vocab)} vocab\n")

    print(f"training dual-head model (LM + Ada role, {STEPS} steps)...")
    p = M.train_dual(ids, rids, len(vocab), N_ROLE, T=T, C=C, steps=STEPS, seed=SEED)
    print()

    print("=" * 76)
    print("1) grounded decode — every content word must be RETURNED by the entity record")
    print("=" * 76)
    for prompt in (["erin", "lives", "in"], ["helium", "is", "a"], ["gold", "looks"],
                   ["jack", "works", "as", "a"], ["erin", "is", "from"]):
        words, events, asserted = hdc_decode(p, mem, stoi, itos, prompt)
        print_trace(words, events, asserted)

    print("=" * 76)
    print("2) refusal — force an ungrounded continuation by removing the true filler")
    print("   from the entity's record; the gate must refuse, not hallucinate")
    print("=" * 76)
    # surgically delete erin's location from the distilled memory (as if never learned)
    import copy
    mem2 = copy.deepcopy(mem)
    mem2.entity_facts["erin"] = [(r, f) for r, f in mem2.entity_facts["erin"]
                                 if r != "spatial.location"]
    from hdc import bind, bundle
    mem2.records["erin"] = bundle([bind(mem2.enc.role_vec(r), mem2.enc.filler_vec(f))
                                   for r, f in mem2.entity_facts["erin"]])
    words, events, asserted = hdc_decode(p, mem2, stoi, itos, ["erin", "lives", "in"])
    print("   (erin's spatial.location fact deleted from memory)")
    print_trace(words, events, asserted)
