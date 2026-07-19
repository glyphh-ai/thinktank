"""Meter demo backend: POST /read {"text": ...} -> named-axis readings.

Loads frozen Qwen2.5-0.5B once, plus the v12 wide-seed axis bank (built on
first run if wide_bank.npz is missing). A text's vector = mean residual
state over its tokens at layer 12, frozen-transformed, projected on the
named axes, z-scored with the frozen tables.

Honesty note (also shown in the UI): the axes are certified on the
word-level protocol; whole-text mean-pooling is a demo-grade extension,
not a certified instrument (that validation is Phase 0 of
products/SPEC_semantic_gateway.md).

Run:  pip install fastapi uvicorn && python server.py   (port 8787)
"""
import os
import sys

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "version_12_wide_seed"))
import wideband  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B"
LAYER = 12
DISPLAY_AXES = [
    "good-bad", "calm-angry", "friendly-hostile", "safe-dangerous",
    "polite-rude", "formal-casual", "true-false", "honest-dishonest",
    "happy-sad", "obey-rebel", "kind-cruel", "war-peace",
]

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

print("loading bank...")
if not os.path.exists(wideband.BANK_NPZ):
    wideband.build_bank()
BANK = wideband.load_bank()
NAMES = [str(n) for n in BANK["names"]]
IDX = [NAMES.index(n) for n in DISPLAY_AXES]

print("loading model...")
TOK = AutoTokenizer.from_pretrained(MODEL)
LM = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
LM.eval()


class ReadReq(BaseModel):
    text: str


@app.post("/read")
def read(req: ReadReq):
    text = req.text.strip()[:2000]
    if not text:
        return {"axes": []}
    with torch.no_grad():
        enc = TOK(text, return_tensors="pt", truncation=True, max_length=256)
        hs = LM(**enc, output_hidden_states=True).hidden_states[LAYER][0]
    # per-token readings, peak-reported: the strongest token reading per axis
    # (mean-pooling a whole text dilutes every needle toward zero; the peak is
    # "the most <axis> moment in this text" — demo-grade, labeled in the UI)
    V = hs.numpy() - BANK["g09"]
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    Z = (V @ BANK["axes"].T - BANK["mu"]) / BANK["sd"]  # [tokens, axes]
    peak_idx = np.argmax(np.abs(Z), axis=0)
    z = Z[peak_idx, np.arange(Z.shape[1])]
    toks = TOK.convert_ids_to_tokens(enc["input_ids"][0])
    out = []
    for name, i in zip(DISPLAY_AXES, IDX):
        zi = float(z[i])
        committed = abs(zi) >= float(BANK["tau"][i])
        out.append({"axis": name, "z": round(zi, 2),
                    "bit": (1 if zi > 0 else -1) if committed else 0,
                    "at": toks[int(peak_idx[i])].replace("Ġ", "")})
    # also surface the 5 largest |z| outside the display set
    rest = [(NAMES[i], float(z[i])) for i in range(len(NAMES)) if i not in IDX]
    rest.sort(key=lambda t: -abs(t[1]))
    extremes = [{"axis": n, "z": round(zi, 2)} for n, zi in rest[:5]]
    return {"axes": out, "extremes": extremes, "tokens": int(enc["input_ids"].shape[1])}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8787)
