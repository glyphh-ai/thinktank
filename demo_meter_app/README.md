# demo_meter_app — the semantic meters, live

A two-piece demo of the v12 instrument bank: paste any text, watch twelve
named needles move, each labeled with the token that moved it most.

```
"The troops advanced at dawn; by nightfall the village was burning."
  good ────────────█──── bad      −1.7σ @burning
  safe ───────────█───── dangerous −1.7σ @burning
```

- `server.py` — FastAPI reader: frozen Qwen2.5-0.5B + the v12 wide-seed
  axis bank (auto-built from repo artifacts if missing). POST `/read`
  returns z-scores, ternary bits (abstention shown as `·off`), and the
  peak token per axis. Deterministic: same text, same needles, forever.
- `web/` — minimal Next.js app (no CSS framework, one page) rendering the
  meters.

## Run

```
pip install fastapi uvicorn
python server.py                 # port 8787 (loads model ~1 min)
cd web && npm install && npm run dev   # http://localhost:3000
```

## Honesty labels (also shown in the UI)

The axes are certified on the word-level protocol (human benchmarks —
see `paper/`). Whole-text readings here use **per-token peak reporting**
("the most *dangerous* moment in this text"), which is demo-grade: livelier
and more interpretable than mean-pooling, but not yet a certified
instrument. Certifying sentence-level reading is Phase 0 of
`products/SPEC_semantic_gateway.md`.
