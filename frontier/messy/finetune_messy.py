"""
finetune_messy.py — the in-loop arm of the messy test: distill Ada's enricher.

The messy task is GENERATION (prose -> Ada-slot JSON), not tagging. So "in-loop"
here = LoRA fine-tune Qwen3 to emit the schema given messy text, trained on
(gold.text -> gold.slots). Then compare, on a held-out split:

  - zero-shot  : base Qwen3, prompted (no fine-tune)
  - in-loop    : the fine-tuned model
  - ceiling    : Ada's own enricher (gold = 1.0 by definition)

scored by slot-F1 vs gold.

PRE-REGISTERED BAR (lock before running):
  in-loop slot-F1 >= zero-shot + 0.10  AND  in-loop >= 0.60

GPU only (Colab). Not run in the CPU sandbox.

Run:  python finetune_messy.py --gold gold.jsonl --model Qwen/Qwen3-8B --steps 600
"""

import json, argparse, random
from eval_messy import slot_f1, PROMPT      # reuse the validated metric + prompt

BAR_DELTA, BAR_ABS = 0.10, 0.60


def target_json(slots):
    return json.dumps(slots, separators=(",", ":"))


def make_examples(tok, rows, max_in=900, max_tgt=200):
    exs = []
    for r in rows:
        p = tok(PROMPT.replace("{t}", r["text"][:1200]), truncation=True, max_length=max_in)["input_ids"]
        t = tok(target_json(r["slots"]), truncation=True, max_length=max_tgt)["input_ids"] + [tok.eos_token_id]
        ids = p + t
        labels = [-100] * len(p) + t            # train only on the JSON target
        exs.append({"ids": ids, "labels": labels})
    return exs


def collate(batch, pad):
    import torch
    T = max(len(b["ids"]) for b in batch)
    ids = torch.tensor([b["ids"] + [pad] * (T - len(b["ids"])) for b in batch])
    lab = torch.tensor([b["labels"] + [-100] * (T - len(b["labels"])) for b in batch])
    am = (ids != pad).long()
    return ids, am, lab


def generate_slots(model, tok, text, dev):
    import torch
    msg = PROMPT.replace("{t}", text[:1200])
    enc = tok(msg, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=200, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    gen = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    try:
        return json.loads(gen[gen.find("{"):gen.rfind("}") + 1])
    except Exception:
        return {}


def eval_split(model, tok, rows, dev):
    return sum(slot_f1(generate_slots(model, tok, r["text"], dev), r["slots"])[2]
               for r in rows) / max(1, len(rows))


def main(gold, model_name, steps, lr):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [json.loads(l) for l in open(gold)]
    random.Random(0).shuffle(rows)
    k = int(0.85 * len(rows)); tr, va = rows[:k], rows[k:]
    print(f"{len(rows)} gold pairs -> train {len(tr)} / val {len(va)}  | {model_name}")

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16) if dev == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=qcfg, device_map="auto" if dev == "cuda" else None,
        torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32)

    # 1) zero-shot baseline (before any fine-tuning)
    zs = eval_split(model, tok, va, dev)
    print(f"[zero-shot] slot-F1 vs Ada gold = {zs:.3f}")

    # 2) in-loop: LoRA fine-tune to emit the schema
    if dev == "cuda": model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    exs = make_examples(tok, tr)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    for s in range(1, steps + 1):
        b = [exs[random.randrange(len(exs))] for _ in range(4)]
        ids, am, lab = (t.to(dev) for t in collate(b, tok.pad_token_id))
        out = model(input_ids=ids, attention_mask=am, labels=lab)
        out.loss.backward(); opt.step(); opt.zero_grad()
        if s % 100 == 0 or s == 1:
            print(f"step {s:4d}  loss {out.loss.item():.3f}")

    model.eval()
    il = eval_split(model, tok, va, dev)

    print(f"\n=== MESSY TEST (slot-F1 vs Ada enricher gold) ===")
    print(f"  zero-shot Qwen3 : {zs:.3f}")
    print(f"  in-loop  Qwen3  : {il:.3f}")
    print(f"  Ada enricher    : 1.000  (ceiling/gold)")
    ok = (il >= zs + BAR_DELTA) and (il >= BAR_ABS)
    print(f"\nPRE-REGISTERED BAR (in-loop >= zero-shot+{BAR_DELTA} and >= {BAR_ABS}): "
          f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="gold.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    main(a.gold, a.model, a.steps, a.lr)
