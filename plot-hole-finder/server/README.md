# plot-hole-finder server

Python backend. This commit lays the **storage bones** for the SaaS: the data
model (auth, billing, projects, the extracted world model, results, usage,
vectors) plus swappable infra for blobs and email. Service layer (FastAPI routes,
auth/JWT logic, Stripe handlers, the analysis pipeline) sits on top of this next.

Runs entirely on **SQLite + local disk + console email** in dev — no Postgres,
R2, or SMTP required. Swaps to **Postgres/pgvector + R2 + SMTP** in prod via env.

## Layout

```
app/
  config.py            env-driven settings, dev-safe defaults
  db.py                engine/session; create_all (dev) / Alembic (prod)
  plans.py             FREE / BASIC / PRO catalog + limits
  models/
    base.py            Base, mixins, portable Embedding + Bipolar column types
    user.py            User, VerificationToken (email PIN / reset), RefreshToken
    billing.py         Subscription, StripeEvent (idempotent webhook log)
    project.py         Project, Document (blob-backed)
    narrative.py       Entity, Fact, AnalysisRun, Issue, IssueEvidence
    usage.py           UsageEvent ledger + sum_usage()
    vectors.py         PassageEmbedding (pgvector), HypervectorRecord (glyphh)
  storage/
    blob.py            BlobStore: LocalDiskBlobStore (dev) / R2BlobStore (prod)
    email.py           EmailSender: Console / File (dev) / SMTP (prod) + templates
tests/
  test_storage_bones.py
```

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest -q
```

## Vector storage (two stores, by design)

- **RAG embeddings** → `PassageEmbedding.embedding` uses pgvector from day one:
  a real `vector(1024)` column + ANN index on Postgres, transparently stored as
  float32 bytes on SQLite so dev needs no extension.
- **glyphh HDC hypervectors** → `HypervectorRecord.data` is int8 bytes
  (D≈10k, bipolar). These are rehydrated into the in-process cortex, never
  ANN-queried in SQL, so they are intentionally *not* pgvector.

## Dev → prod via env

| Concern | Dev default | Prod |
| --- | --- | --- |
| DB | `sqlite+pysqlite:///./var/app.db` | `postgresql+psycopg://…` (`DATABASE_URL`) |
| Blobs | local disk (`./var/blobs`) | R2 (`BLOB_BACKEND=r2`, `R2_*`) |
| Email | console (prints PIN) | Resend (`EMAIL_BACKEND=resend`, `RESEND_API_KEY`) |

### Email backends

`console` (default, dev) prints the PIN to the log; `file` writes `.eml`s;
`smtp` targets local Mailpit/MailHog or any SMTP host; `resend` uses the Resend
HTTP API. For Resend, set `EMAIL_BACKEND=resend`, `RESEND_API_KEY`, and
`EMAIL_FROM` to an address on a domain you've verified in Resend (or
`onboarding@resend.dev` for quick tests).
| Billing | disabled | Stripe (`STRIPE_*`) |

## Plans

| Tier | Projects | Words/project | Find | Fix |
| --- | --- | --- | --- | --- |
| Free | 1 | 3,000 | ✅ | ❌ |
| Basic | 20 | 300,000 | ✅ | ❌ |
| Pro | 200 | 600,000 | ✅ | ✅ (50/mo, included) |
