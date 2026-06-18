"""
finetune_qwen.py — Arm B: Ada IN the loop on a real Qwen3 (LoRA + role head).

Fine-tunes Qwen3 with the SAME dual objective validated in smoke_test.py, but on
a real model and real BPE tokenization:
  loss = causal_LM_loss  +  lambda * Ada_role_token_loss
A small token-classification head over the last hidden states predicts each
token's Ada slot; word-level role labels are aligned to BPE subwords via offset
mapping. Only LoRA adapters + the role head train.

>>> NOT runnable in the CPU sandbox — validated only on a Colab GPU. The pipeline
    LOGIC is what smoke_test.py verifies; this is the same logic at scale. <<<

Run (on Colab GPU, after prepare_data.py):
    python finetune_qwen.py --model Qwen/Qwen3-8B --steps 1500
"""

import os, json, argparse, math
import torch, torch.nn as nn
from prepare_data import ALL_ROLES

ROLES = ["O"] + ALL_ROLES
RID = {r: i for i, r in enumerate(ROLES)}


def pick_model():
    """Biggest Qwen3 that fits the detected GPU (QLoRA 4-bit)."""
    if not torch.cuda.is_available():
        return "Qwen/Qwen3-0.6B"
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gb >= 38:  return "Qwen/Qwen3-8B"     # A100 40GB
    if gb >= 22:  return "Qwen/Qwen3-4B"     # L4 24GB
    return "Qwen/Qwen3-1.7B"                  # T4 16GB


def build_examples(tokenizer, path="data/facts.jsonl", n=20000, T=64):
    """Tokenize each fact; align word roles -> subword role labels via offsets."""
    exs = []
    for i, line in enumerate(open(path)):
        if i >= n: break
        rec = json.loads(line)
        wr = rec["word_roles"]
        text = " ".join(w for w, _ in wr)
        enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=T)
        # map char span -> word index -> role
        spans, c = [], 0
        for w, r in wr:
            spans.append((c, c + len(w), RID[r])); c += len(w) + 1
        roles = []
        for (a, b) in enc["offset_mapping"]:
            if a == b:           # special token
                roles.append(-100); continue
            rid = -100
            for (sa, sb, rr) in spans:
                if a >= sa and a < sb:
                    rid = rr; break
            roles.append(rid)
        exs.append({"input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"], "roles": roles})
    return exs


class AdaQwen(nn.Module):
    """Qwen3 (LoRA) + a token-classification head for Ada roles."""
    def __init__(self, base, hidden, n_roles, lam=0.5):
        super().__init__()
        self.base = base
        self.role_head = nn.Linear(hidden, n_roles)
        self.lam = lam

    def forward(self, input_ids, attention_mask, labels=None, role_labels=None):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        lm_logits = out.logits
        role_logits = self.role_head(out.hidden_states[-1].float())
        loss = lm_loss = role_loss = None
        if labels is not None:
            ce = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = ce(lm_logits[:, :-1].reshape(-1, lm_logits.size(-1)),
                         labels[:, 1:].reshape(-1))
            role_loss = ce(role_logits.reshape(-1, role_logits.size(-1)),
                           role_labels.reshape(-1))
            loss = lm_loss + self.lam * role_loss
        return loss, lm_logits, role_logits, lm_loss, role_loss


def collate(batch, pad_id):
    T = max(len(b["input_ids"]) for b in batch)
    def pad(key, fill): return torch.tensor([b[key] + [fill]*(T-len(b[key])) for b in batch])
    ids = pad("input_ids", pad_id); am = pad("attention_mask", 0); roles = pad("roles", -100)
    labels = ids.clone(); labels[am == 0] = -100
    return ids, am, roles, labels


def main(model_name, steps, lam, n, lr):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = model_name or pick_model()
    print(f"[device {dev}] model {model_name}  lambda {lam}")

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16) if dev == "cuda" else None
    base = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=qcfg, device_map="auto" if dev == "cuda" else None,
        torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    if dev == "cuda": base = prepare_model_for_kbit_training(base)
    base = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    hidden = base.config.hidden_size
    model = AdaQwen(base, hidden, len(ROLES), lam=lam).to(dev)
    model.role_head.to(dev).to(torch.float32)

    exs = build_examples(tok, n=n)
    k = int(0.95*len(exs)); tr, va = exs[:k], exs[k:]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    import random
    model.train()
    for s in range(1, steps+1):
        batch = [tr[random.randrange(len(tr))] for _ in range(8)]
        ids, am, roles, labels = (t.to(dev) for t in collate(batch, tok.pad_token_id))
        loss, *_ , lml, rl = model(ids, am, labels, roles)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 0 or s == 1:
            print(f"step {s:4d}  loss {loss.item():.3f}  lm {lml.item():.3f}  role {rl.item():.3f}")

    # eval role-tag accuracy on val
    model.eval(); ok = tot = 0
    with torch.no_grad():
        for i in range(0, len(va), 8):
            b = va[i:i+8]
            ids, am, roles, labels = (t.to(dev) for t in collate(b, tok.pad_token_id))
            _, _, rlog, _, _ = model(ids, am, labels, roles)
            pred = rlog.argmax(-1); m = roles != -100
            ok += (pred[m] == roles[m]).sum().item(); tot += m.sum().item()
    print(f"\nval Ada role-tag accuracy: {ok/max(1,tot):.1%}  (lambda={lam})")
    print("Compare lambda>0 vs lambda=0: does the role objective hold at frontier scale,")
    print("and does it stay free on the LM loss?")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    main(a.model, a.steps, a.lam, a.n, a.lr)
