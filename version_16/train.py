"""
Train — AdamW, warmup + cosine decay, periodic eval / checkpoint / sample
=======================================================================
Run:  python -u train.py            (resumes from ckpt.pt if present)
Out:  ckpt.pt
"""

import math
import time
from pathlib import Path

import numpy as np
import torch

from data import DATA, decode, load_vocab
from model import GPT, Config

STEPS = 20_000
BATCH = 32
LR = 1e-3
MIN_LR = 1e-4
WARMUP = 500
WEIGHT_DECAY = 0.1
EVAL_EVERY = 500
EVAL_ITERS = 20
CKPT = Path(__file__).parent / "ckpt.pt"

torch.manual_seed(0)
words, stoi = load_vocab()
cfg = Config(vocab_size=len(words))
train_ids = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
val_ids = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")


def get_batch(split):
    src = train_ids if split == "train" else val_ids
    ix = np.random.randint(0, len(src) - cfg.block_size - 1, BATCH)
    x = np.stack([src[i:i + cfg.block_size] for i in ix]).astype(np.int64)
    y = np.stack([src[i + 1:i + 1 + cfg.block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    t = (step - WARMUP) / max(1, STEPS - WARMUP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses, accs = [], []
        for _ in range(EVAL_ITERS):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses.append(loss.item())
            accs.append((logits.argmax(-1) == y).float().mean().item())
        out[split] = (np.mean(losses), np.mean(accs))
    model.train()
    return out


def sample(model, n=80):
    model.eval()
    idx = torch.tensor([[stoi["<eot>"]]])
    text = decode(model.generate(idx, n)[0, 1:].tolist(), words)
    model.train()
    return text


if __name__ == "__main__":
    model = GPT(cfg)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=LR, betas=(0.9, 0.95))
    start = 0
    if CKPT.exists():
        ck = torch.load(CKPT)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print(f"resumed from step {start}")

    print(f"{cfg}\nparams {model.n_params():,} | train tokens {len(train_ids):,}")
    t0 = time.time()
    for step in range(start, STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            dt = (time.time() - t0) / (step - start + 1)
            print(f"step {step:6d} | loss {loss.item():.3f} | lr {lr_at(step):.2e} | {dt:.2f}s/step")
        if (step + 1) % EVAL_EVERY == 0 or step + 1 == STEPS:
            ev = evaluate(model)
            print(f"== step {step + 1} | train {ev['train'][0]:.3f} ({ev['train'][1]:.1%})"
                  f" | val {ev['val'][0]:.3f} ({ev['val'][1]:.1%})")
            print("   " + sample(model))
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1, "cfg": cfg.__dict__}, CKPT)
