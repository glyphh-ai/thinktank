"""
eval_messy.py — the real-world messy test: extract Ada slots from MESSY prose.

Unlike the clean task (token-tag a value that appears verbatim), here the slot
value is often paraphrased/implicit in the prose -> the model must GENERATE the
structured slots, i.e. be Ada's enricher. We score generated slots vs gold
(Ada's enricher output) by slot-level F1.

Systems compared:
  - zero-shot Qwen3   : prompt the base model to emit the schema (no fine-tune)
  - in-loop Qwen3     : the model fine-tuned to emit Ada slots (frontier/finetune_qwen)
The question: does in-loop training make the model a BETTER enricher on messy
text than prompting alone — and how close to gold?

The slot-F1 metric is validated on CPU here (self-test). The Qwen3 generations
need a GPU (Colab).

Run (CPU self-test of the metric):   python eval_messy.py --selftest
Run (GPU eval, after gold.jsonl):    python eval_messy.py --gold gold.jsonl
"""

import json, argparse


def flat(slots):
    """{layer:{role:value}} -> set of 'layer.role=value' (lowercased)."""
    out = set()
    for layer, roles in (slots or {}).items():
        for role, val in (roles or {}).items():
            if val:
                out.add(f"{layer}.{role}={str(val).strip().lower()}")
    return out


def slot_f1(pred, gold):
    P, G = flat(pred), flat(gold)
    if not P and not G:
        return 1.0, 1.0, 1.0
    tp = len(P & G)
    prec = tp / len(P) if P else 0.0
    rec = tp / len(G) if G else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def selftest():
    gold = {"entity": {"name": "Caffeine", "kind": "stimulant"},
            "relational": {"predicate": "blocks", "object": "adenosine"}}
    perfect = gold
    partial = {"entity": {"name": "caffeine", "kind": "drug"},          # kind wrong
               "relational": {"predicate": "blocks", "object": "adenosine"}}
    empty = {}
    for name, pred in (("perfect", perfect), ("partial", partial), ("empty", empty)):
        print(f"  {name:8} P/R/F1 = %.2f %.2f %.2f" % slot_f1(pred, gold))
    assert slot_f1(perfect, gold)[2] == 1.0
    assert 0 < slot_f1(partial, gold)[2] < 1.0
    assert slot_f1(empty, gold)[2] == 0.0
    print("slot-F1 metric self-test PASSED.")


# ---- GPU eval (Colab): prompt/generate Ada slots from messy text ----
PROMPT = (
    "Extract a fixed-schema record from the text. Output JSON with only these "
    "layers/roles when present: entity{name,kind,subkind}, perceptual{color,"
    "size,shape,...}, spatial{location,origin}, temporal{time,age,era}, "
    "relational{predicate,object}, quantitative{count,magnitude,unit}. "
    "Text:\n{t}\nJSON:"
)

def gpu_eval(gold_path, model_name):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                             torch_dtype=torch.bfloat16)
    rows = [json.loads(l) for l in open(gold_path)]
    f1s = []
    for r in rows:
        msg = PROMPT.replace("{t}", r["text"][:1200])
        ids = tok(msg, return_tensors="pt").to(m.device)
        out = m.generate(**ids, max_new_tokens=200, do_sample=False)
        gen = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        try:
            pred = json.loads(gen[gen.find("{"):gen.rfind("}") + 1])
        except Exception:
            pred = {}
        f1s.append(slot_f1(pred, r["slots"])[2])
    print(f"zero-shot {model_name}: mean slot-F1 vs Ada gold = {sum(f1s)/len(f1s):.3f}  (n={len(f1s)})")
    print("(compare to an in-loop fine-tuned model trained to emit the schema)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gold", default="gold.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        gpu_eval(a.gold, a.model)
