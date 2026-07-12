# Migration map — retire `adaL` as a repo, fold it into `glyphh`

Hand-off for an agent working in the glyphh codebase. Verified against the actual
source trees (2026-07-12), not assumed.

## Corrected topology (the important part)

`adaL` and `glyphh` do **not** share primitives. They are two related subsystems
built on different math, meeting at one seam:

| | symbolic vocabulary | vector system | store | ingestion |
|---|---|---|---|---|
| **Ada** (`adaL`) | **NSM semantic primes** (`ada/rotor/primes.py`: ABOVE, BECAUSE, DIE, KNOW, LIVE… + `decompose`/`validate`) | **semantic embeddings** (nomic float vectors, `ada/rotor/embed.py`) | SQLAlchemy / `DATABASE_URL` (rotor_turns, memory) | **front half**: enrichers `ada/encoder/{llm,local}_enricher.py` — `enrich(text)->StructuredFact` |
| **Glyph** (`glyphh` + `glyphh-runtime`) | **per-model role slots** (`roles_config` JSON, dynamic — `entity.name`, `spatial.location`) | **bipolar HDC** (`glyphh.encoder`/`glyphh.vector`: bind/bundle/cortex) | **Postgres** (`Glyph`/`Segment`/`Embedding`, BYTEA + pgvector) | **back half**: `ingest_concepts(List[ConceptInput])` -> encode -> both structures -> store |

Two facts that drive every decision below:
- **The HDC primitives are NOT in Ada.** Ada's vectors are *semantic embeddings*
  (nomic), not bipolar HDC. These are complementary roles (embeddings = what a
  fact *means*; HDC = how it's *bound/structured*) — keep BOTH, don't unify.
- **Primes ≠ roles.** Ada decomposes into fixed NSM primes for *recall grouping*;
  glyphh binds into per-model role slots for *structure*. Different ontologies,
  different jobs. Don't force one into the other.

## The clean seam (this part IS a mechanical merge)

```
raw text ──[Ada enricher]──▶ StructuredFact ≈ ConceptInput ──[glyphh ingest]──▶ semantic + cortex + segments ──▶ store
           (front half, move)                 (already the boundary)            (back half, exists)
```

`glyphh` already owns everything right of `ConceptInput`. Moving Ada's enricher in
front of `/ingest` makes ingestion end-to-end (raw -> both structures) in one repo.

## Module-by-module disposition

**MOVE (clean):**
- `ada/encoder/llm_enricher.py`, `ada/encoder/local_enricher.py` → `glyphh/enrich/`.
  Wire `enrich(text) -> ConceptInput` in front of `/ingest`; add a `raw=True` mode
  to the ingest route. Keep the enricher separable so it can also run **offline/batch**
  (the doc's "schema-on-write compiler runs offline").

**MOVE as a DISTINCT module (do not merge into HDC):**
- `ada/rotor/*` (primes decomposition + nomic embeddings + prime-based recall +
  short-term memory) → `glyphh/recall/`. This is a capability glyphh lacks
  (semantic conversational recall). It stays its own subsystem: NSM primes +
  embeddings, separate from the HDC encoder. (This is also the semantic layer the
  `schema_guard` experiment showed you need — embeddings, not HDC.)

**RECONCILE (the real work / risk):**
- **Stores.** Ada (SQLAlchemy, likely sqlite-default) vs glyphh Postgres
  (`Glyph`/`Segment`/`Embedding`, pgvector). Pick Postgres as authoritative for the
  structural lattice; migrate Ada's `rotor_turns`/memory tables into it (or keep a
  separate short-term table). Map every column; write an alembic migration.
- **Symbolic vocab contract.** Decide how NSM primes (recall) relate to per-model
  `roles_config` (binding). Simplest: they stay independent — primes index recall,
  roles define binding — with the **enricher** as the one place that emits both a
  role-structured `ConceptInput` (for HDC) and prime tags (for recall).
- **space_id invariant.** Keep `sha256(vector_dim, seed, roles_config)`; after the
  merge the enricher and encoder share one `roles_config` per model, so drift can
  only come from intended config changes. (See `grounding.py::assert_space`.)

**DECIDE ownership (overlaps another product):**
- `ada/rotor/proxy.py` is an Anthropic API **gateway/proxy** (routes with Ada's
  stored key). This overlaps the `glyphh-apps` gateway product. Pick a single
  owner; don't ship two proxies.

**EVALUATE / likely archive:**
- `ada/tui`, `ada/skill`, `ada/domains`, `ada/cognitive` (beyond the enricher),
  `benchmarks/` → move what's load-bearing, archive the rest. Benchmarks worth
  keeping (Phase-1 extraction, Phase-3 NL query) go to `glyphh/benchmarks/`.

**DELETE (drift risk, incl. our own):**
- Vendored copies of the schema/HDC math — including this prototype's `hdc.py` and
  the vendored `UNIVERSAL_SCHEMA` in `corpus.py`. After the merge everything imports
  the canonical `glyphh` encoder + schema, nothing vendors.

## Ordered checklist

1. Land the enricher in `glyphh/enrich/`; add `raw` mode to `/ingest`
   (`enrich -> ConceptInput -> ingest_concepts`). Ship the end-to-end path first.
2. Move `ada/rotor/*` → `glyphh/recall/` as a separate subsystem (primes +
   embeddings). Do NOT touch the HDC encoder.
3. Store reconciliation: choose Postgres authoritative; alembic-migrate Ada tables.
4. Decide gateway ownership (proxy.py vs glyphh-apps).
5. Point everything at one `roles_config`-per-model; delete vendored schema/HDC.
6. Move the load-bearing benchmarks; archive tui/skill/domains.
7. Retire the `adaL` repo. Keep **"Ada"** as the *product/substrate name* for the
   enrichment + schema + recall layer living inside glyphh — the repo dies, the
   noun doesn't.

## Open decisions for you (not mine to make)

- **Keep both vector systems?** Recommended yes (embeddings=meaning, HDC=structure —
  proven complementary). Confirm.
- **Recall subsystem** (rotor) — product feature or legacy? If legacy gateway-era,
  archive instead of move.
- **Gateway** — glyphh owns it, or stays in glyphh-apps?
- **Store** — single Postgres, or Postgres (structural) + a light local store
  (short-term recall)?
