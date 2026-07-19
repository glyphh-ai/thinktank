# SPEC — Glyphh Gate: a semantic gateway proxy (firewall + introspective routing)

**Handoff document for a coding agent.** Read this whole file before writing
code. The science this product stands on lives in this repo (versions 07–12
plus `paper/`); reuse it, don't reinvent it.

## 0. One paragraph of context

Versions 07–12 of this repo built and externally validated a **semantic
instrument bank**: 242 named contrast axes (good–bad, obey–rebel,
formal–casual, …) read from a frozen Qwen2.5-0.5B by dot product — no
training, deterministic, ~milliseconds on CPU, benchmarked against human
ground truth (SimLex-999 ρ=0.497 ≥ its own substrate; zero-shot human
valence ρ=0.57). Readings quantize to ternary named bits (abstention
certified) and store/query in a bipolar hypervector memory (bind/bundle,
threshold-unbind — see `version_06_hdc_forward/` and `version_10`).
Glyphh Gate puts that instrument inline on LLM traffic.

## 1. What the product is

An OpenAI-compatible proxy (`gate` CLI, REPL first, web UI later) that, for
every request and response passing through it:

1. **Reads** the text on K named axes (semantic telemetry, logged).
2. **Screens** it against a threat memory (prompt-injection firewall).
3. **Routes** it by rule (semantic routing: which upstream model, which
   system prompt, block/flag/escalate).
4. **Explains** every decision with a receipt: the named bits and matched
   patterns that fired. No black-box scores anywhere.

Differentiators no incumbent gateway has: deterministic per-request
semantics at wire speed (no judge-model API calls), bit-level auditability,
on-prem/air-gapped, and an HDC threat memory that names *which* known
attack family a prompt resembles.

## 2. Honest engineering note: the firewall hypothesis is PARTLY unproven

Read this before building Phase 2. Two claims must be separated:

- **Known-pattern matching (strong ground).** Encoding a corpus of known
  injection/jailbreak prompts as barcodes/hypervectors and flagging inputs
  by similarity — with the matched exemplar cited in the receipt — is a
  direct application of validated machinery (v10 T2/T3, v12 similarity).
  Expect this to work and to be fast, auditable, and updatable by appending
  patterns (no retraining).
- **Novel-attack detection via semantic meters (hypothesis).** Injection is
  about *imperative intent and boundary violation* ("ignore your
  instructions"), not valence. The existing 242 axes were never validated
  for this. New axes must be mined for instruction-register contrasts
  (command–statement, obey–rebel, system–user voice, roleplay–direct,
  meta–object language) and **validated before being trusted**. Paraphrased
  and encoded attacks (base64, translation, payload-splitting) may defeat
  lexical-ish readings entirely. Do not claim novel-attack detection until
  Phase 0 gates pass. Firewalls die on false positives: the gate metric is
  **FPR at fixed recall**, not accuracy.

Phase 0 (validation, ~days, all offline) is therefore mandatory before any
firewall marketing claim:

- Assemble eval sets: `deepset/prompt-injections`, jailbreak datasets on
  HuggingFace (e.g. jackhhao/jailbreak-classification, JailbreakBench
  artifacts), plus a benign corpus that *looks* spicy (security blogs, IT
  helpdesk tickets, fiction with commands) — benign-but-imperative text is
  the false-positive trap.
- Pre-registered gates (thinktank house style — commit gates before
  results): known-pattern memory ≥0.95 recall on held-out variants of seen
  families at ≤1% FPR; meters+memory ensemble on *unseen* families:
  report honestly, gate at ≥0.70 recall at ≤2% FPR to call it "novel-attack
  assist"; below that, ship pattern-matching only and say so.
- Baselines to beat or complement: Lakera Guard (closed), Meta
  PromptGuard/LlamaGuard (open — also candidate ensemble members ON the
  gateway, they're small classifiers; our edge is receipts + determinism +
  threat-memory updatability, not necessarily raw AUC).

## 3. Architecture

```
client ──HTTP──▶ gate proxy ──▶ upstream LLM(s) (OpenAI-compatible)
                   │
                   ├─ reader   : frozen Qwen2.5-0.5B, layer 12, axes bank
                   │             (port of version_12_wide_seed/wideband.py)
                   ├─ threat   : HDC memory of attack barcodes
                   │             (port of version_10 step4 lattice, D=16384)
                   ├─ rules    : YAML rule engine over readings + matches
                   ├─ ledger   : append-only JSONL of receipts
                   └─ repl/cli : `gate` command (below)
```

- **Reader**: text → residual states at layer 12 → mean-pool token vectors →
  frozen transform → z-scores on K axes → ternary bits. Word-level protocol
  is validated; sentence-level pooling is v13-grade — validate pooling
  choice in Phase 0 (compare mean-pool vs last-token vs max-|z| pooling on
  the eval sets; pick by gate, not vibes).
- **Threat memory**: each known attack → barcode → bound record; query =
  ternary cosine / unbind. Store family labels; receipts cite
  `matched: DAN-family exemplar #4471 (cos 0.83)`.
- **Rules** (YAML): `when: {axis: obey-rebel, z: "< -1.5"} then: {route:
  guarded-model, tag: imperative}` / `when: {threat.cos: "> 0.75"} then:
  {action: block, receipt: true}`. Rules compose; first-match or all-match
  modes; dry-run mode logs what *would* have fired (essential for tuning
  FPR on live traffic before enforcement).
- **Perf budget**: reader ≤50ms p50 on CPU for ≤512 tokens (batch=1);
  proxy adds ≤80ms p50 end-to-end. If missed, quantize the 0.5B (int8) or
  cache by text hash.

## 4. CLI/REPL UX (v0 deliverable)

```
gate serve --config gate.yaml --port 8080     # OpenAI-compatible proxy
gate read "ignore previous instructions..."   # one-shot: readings + verdict
gate repl                                     # interactive: paste → receipt
gate threats add attacks.jsonl --family DAN   # grow the memory, no retrain
gate threats test evalset.jsonl               # recall/FPR report vs gates
gate rules test --replay traffic.jsonl        # dry-run rule tuning
gate bench                                    # latency budget check
```

REPL output style (the receipt IS the product — make it beautiful):

```
▸ verdict: BLOCK  (rule: injection-guard)
  threat  : DAN-family #4471  cos 0.83   [strong]
  meters  : obey-rebel −2.1σ  meta-object +1.8σ  formal-casual −0.4σ
  route   : (blocked; would have gone to gpt-4o-mini)
  receipt : ledger#000482  (deterministic — same input always yields this)
```

## 5. Build phases

| phase | deliverable | gate |
|---|---|---|
| 0 | validation harness + eval report | pre-registered gates in §2 |
| 1 | reader service + `gate read`/`repl` | latency budget; readings reproduce repo numbers on the v12 vocab |
| 2 | threat memory + receipts | Phase 0 known-pattern gate |
| 3 | proxy + rules + ledger + dry-run | passthrough adds ≤80ms p50; rules fire correctly on replayed traffic |
| 4 | packaging (pip/docker), docs | fresh-machine install ≤10 min |
| 5 | web UI (later) | live traffic view + receipt browser; do NOT start before 0–4 ship |

## 6. Tech + reuse map

Python 3.11, FastAPI, PyTorch CPU (int8 optional), numpy. Reuse directly:
extraction protocol (`version_07_prime_basis/step1_vectors.py` pattern),
axes+frozen tables (`version_12_wide_seed/wideband.py`, `pairs.py`), ternary
codes (`version_10_ternary_barcode/ternary.py`), HDC ops
(`version_06_hdc_forward/hdc.py`, v10 `step4_lattice.py`). Keep the model
frozen and the calibration tables versioned artifacts (hash-pinned) — the
determinism claim depends on it.

## 7. Risks, stated plainly

- Novel-attack recall may be mediocre → ship as pattern-firewall + telemetry
  + routing, which is already differentiated; don't overclaim.
- Sentence-pooling may dilute meters → Phase 0 decides pooling; worst case
  per-sentence windowing.
- Encoded payloads (base64 etc.) are out of scope for semantic reading →
  pair with cheap structural detectors (entropy, encoding sniffers) in the
  rules layer; be explicit in docs.
- Incumbents (Lakera, PromptGuard) exist → our wedge is receipts,
  determinism, on-prem, and the updatable named threat memory, not raw AUC.
