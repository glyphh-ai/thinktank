# plot-hole-finder

Analyzes large documents (novels, screenplays, legal/professional documents) and
surfaces **plot holes, continuity errors, and timeline/sequencing problems**, then
helps fix and export them.

## What it finds

- **Continuity errors** — an attribute that changes with no event to explain it,
  an object destroyed then used, a character who dies then reappears.
- **Timeline & sequencing** — effects before causes, knowledge before it's
  possible, broken setup/payoff, out-of-order callbacks.
- **Logical gaps & character inconsistencies** — references to things never
  established, out-of-character behavior.

## Architecture

A **standalone TypeScript Node server** in [`server/`](server/):

- **Fastify** control-plane REST API (auth, projects/upload, jobs, billing) + an
  interactive **agent SSE** endpoint.
- A **true MCP server** (`@modelcontextprotocol/sdk`, Streamable HTTP,
  per-user/project authenticated) exposing the plot-hole toolset at `/mcp`.
- **BullMQ** job queue + dedicated **worker** process so massive documents
  (200k–1M+ tokens) are analyzed/fixed in the background with retries and
  progress — not inside an HTTP request.
- **Redis** (queue + cache/rate-limit) and **Postgres/Drizzle** (durable state:
  users, projects, documents, the entity-indexed memory store, issues, timeline,
  notes).

The detection itself is a **Claude Agent SDK** rolling-memory loop: backend tools
read the document, maintain a retrieval-backed memory (so it scales past the
context window), and record issues / timeline / notes / edits. Permission modes
(ask / plan / auto / bypass) gate document edits.

Detection is Claude's — established by the evals recorded in the project history:
the agent + memory approach beat a bespoke extraction/HDC pipeline, and at >1M
tokens it's the only thing that runs. So the product value is the agent loop, the
tools, scale, ingestion, and the (in-progress) UI.

`docker compose up --build` runs the full stack (Postgres + Redis + server +
worker). See [`server/README.md`](server/README.md) to run it.

> `DESIGN.md` documents the original (HDC/glyphh) design and is kept as historical
> record; that approach was benchmarked out and removed. The Python prototype has
> been deleted in favor of the TypeScript server.
