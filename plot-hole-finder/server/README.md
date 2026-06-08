# plot-hole-finder server

Python/FastAPI backend: auth, billing, project upload, and the analysis pipeline
(extract → glyphh cortex triage → RAG evidence → LLM verify → issues).

Runs **fully offline in dev** (SQLite + local disk + console email + heuristic
extractor + hashing embedder + disabled billing). Flips to **Postgres/pgvector +
R2 + Resend + Claude + Stripe** in prod via env — no code changes.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # optional; sensible dev defaults work without it
python -m pytest -q           # 19 tests, no network needed
uvicorn app.main:app --reload # http://localhost:8000/docs
```

## Configuration (env)

`.env` is auto-loaded (gitignored). Each concern has an offline dev default and a
prod backend selected by an env var:

| Concern | Dev default | Prod | Selector |
| --- | --- | --- | --- |
| DB | SQLite | Postgres/pgvector | `DATABASE_URL` |
| Embeddings | hashing | FastEmbed `bge-small` | `EMBED_BACKEND=fastembed` |
| Extraction/verify | heuristic | Claude | `LLM_BACKEND=claude` + `ANTHROPIC_API_KEY` |
| Email | console | Resend / SMTP | `EMAIL_BACKEND=resend` |
| Blobs | local disk | R2 | `BLOB_BACKEND=r2` |
| Billing | disabled | Stripe | `BILLING_BACKEND=stripe` |
| Cache | in-memory | Redis | `CACHE_BACKEND=redis` + `REDIS_URL` |
| Jobs | inline | external worker | `JOB_RUNNER=external` |
| Migrations | auto (additive) | Alembic | `AUTO_MIGRATE=false` |

## API

| Method | Path | Notes |
| --- | --- | --- |
| POST | `/auth/register` | creates user, emails a 6-digit PIN |
| POST | `/auth/verify-email` | confirm PIN |
| POST | `/auth/resend-pin` | re-send PIN |
| POST | `/auth/login` | → access + refresh JWTs |
| POST | `/auth/refresh` | rotate refresh token |
| POST | `/auth/logout` | revoke refresh token |
| POST | `/auth/forgot-password` / `/auth/reset-password` | emailed reset link |
| GET | `/auth/me` | current user + plan |
| POST | `/projects` | create from pasted text |
| POST | `/projects/upload` | create from file (multipart) |
| GET/DELETE | `/projects` , `/projects/{id}` | list / fetch / delete |
| POST | `/projects/{id}/analyze` | run find pipeline |
| POST | `/projects/{id}/fix` | Pro: suggest fixes |
| GET | `/projects/{id}/issues` / `/runs` | results |
| POST | `/billing/checkout` / `/portal` | Stripe Checkout / portal |
| POST | `/billing/webhook` | Stripe events (idempotent) |

Plan limits (projects, words/doc, fix access) are enforced on create/analyze;
`current_plan()` derives the tier from the user's active subscription.

## Layout

```
app/
  config.py  db.py  plans.py  security.py  text.py  errors.py  deps.py  main.py
  models/      ORM (users, billing, projects, narrative, usage, vectors)
  storage/     blob (local/R2) + email (console/file/smtp/resend)
  embeddings/  Embedder (hash/fastembed), pgvector-aware search
  llm/         Extractor/Verifier (heuristic/claude)
  services/    auth, projects, billing, analysis (the pipeline)
  routers/     auth, projects, analysis, billing
  glyphh_bridge.py   imports the ../engine cortex
tests/         storage bones + full API e2e
```

## The pipeline (`services/analysis.py`)

Detection is **rolling memory** — Claude reads the document in chunks, maintains a
running memory of durable facts/constraints, and flags a chunk when it
contradicts memory with no event explaining it (`verifier.rolling_scan`). This
won the eval bake-off (see `eval/README.md`); the earlier extraction + glyphh
cortex path was more complex and lower-recall and was removed from the product.

1. **Embed passages** — chunk each document small; embed → `PassageEmbedding` (RAG
   evidence index).
2. **Rolling scan** — chunk large; Claude maintains running memory across chunks
   and emits inconsistencies. Scales past the context window (no call sees the
   whole doc).
3. **Record** — each inconsistency → `Issue` + `IssueEvidence` grounded with RAG
   (jump-to-passage).

> At very large scale the running memory needs a retrieval backend (so it isn't
> re-sent whole each chunk) — that's the next engineering step, not a new
> algorithm. Detection quality is Claude's; the product value is scale, ingestion,
> evidence/UX, and domain evals.

## Vectors

- RAG passage embeddings → pgvector `vector(384)` on PG (+ HNSW index), bytes on
  SQLite. (The glyphh cortex is no longer in the detection path.)

## Background jobs

Analysis runs as a **durable, DB-backed job** (`jobs` table = source of truth:
status, progress, attempts, lease, backoff). Never run minutes-long work inside a
request. Three runner modes (`JOB_RUNNER`):

- `inline` — API drains the queue in-request (tests / simple dev)
- `thread` — a daemon thread polls the queue inside the API (dev convenience)
- `external` — API only enqueues; run the worker separately (prod):
  `python -m app.worker`. Scale by running more worker replicas; claiming uses
  `FOR UPDATE SKIP LOCKED` on Postgres for safe concurrency.

`POST /projects/{id}/analyze` returns `202` + a job; poll `GET /jobs/{id}` for
progress, or cancel with `POST /jobs/{id}/cancel`.

## Schema changes

Dev auto-migrates **additive** changes (new tables/columns) on startup
(`AUTO_MIGRATE=true`). Drops/renames/type changes and prod use **Alembic**
(`AUTO_MIGRATE=false`); `python -m app.release` is the deploy hook.

## Docker

From `plot-hole-finder/`:

```bash
docker compose up --build   # API + worker + Postgres/pgvector + Redis
# enable real extraction: LLM_BACKEND=claude ANTHROPIC_API_KEY=... docker compose up
```

API on `:8000`, Postgres on `:5432`, Redis on `:6379`. The worker runs as its own
service (`JOB_RUNNER=external`).

## Deployment

`Procfile` defines `web`, `worker`, and `release` processes for Heroku-style
platforms. See repo root notes for hosting options (Heroku / Railway / Render /
Fly / managed K8s).
