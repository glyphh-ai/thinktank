# plot-hole-finder server

A standalone TypeScript Node server: **Fastify** API + a **true MCP server** +
**BullMQ** workers + the **Claude Agent SDK** rolling-memory loop. Detection is
the agent, so the value is the agent loop, the tools, scale, ingestion, and
(later) the 3-panel UI.

> Status: **server/agent core; UI intentionally not built yet.**

## Components

- **Fastify API** (`src/http`) — control-plane REST: auth, projects/upload,
  job status, billing; plus an interactive **agent SSE** endpoint (`POST /api/agent`).
- **True MCP server** (`src/mcp`) — `@modelcontextprotocol/sdk` over Streamable
  HTTP at `POST /mcp`, **per-user/project authenticated** (Bearer access token +
  `x-plothole-project` header). Any MCP client can call the plot-hole toolset.
- **BullMQ worker** (`src/jobs`, `src/worker.ts`) — runs long analyze/fix passes
  over massive documents in the background, with retries, progress, and lock
  renewal. Enqueued via `POST /api/projects/:id/analyze|fix`.
- **Redis** (`lib/redis.ts`) — queue + cache + rate-limit (in-memory fallback in dev).
- **Postgres / Drizzle** (`lib/db`, `lib/store`) — durable state: users, billing,
  projects, documents, the entity-indexed memory store, issues+evidence,
  timeline, notes.

## The toolset (one definition, two surfaces)

`lib/agent/tool-defs.ts` is the single source of truth. It is adapted to the
Agent SDK in-process tools (`lib/agent/tools.ts`, used by the worker's `runAgent`
loop) **and** registered on the true MCP server (`src/mcp/server.ts`).

| Tool | Purpose |
| --- | --- |
| `read_document` | page through the document (large-doc safe) |
| `remember_facts` / `recall_memory` | retrieval-backed memory across chunks |
| `record_issue` | a plot hole with evidence spans |
| `record_timeline_event` / `get_timeline` | story-time vs narrative-time → sequencing checks |
| `add_note` | flag a span on the document timeline |
| `list_issues` | read findings |
| `apply_edit` | change the document (**gated by permission mode**) |
| `export_document` | return the document for save/export |

### Permission modes (drive the UI's mode switch + button styling later)

| Mode | SDK mode | Document edits |
| --- | --- | --- |
| `ask` | default | require approval (UI round-trip) |
| `plan` | plan | denied (analyze only) |
| `auto` | default | auto-approved |
| `bypass` | default | auto-approved (no checks) |

`lib/agent/permissions.ts` is the authoritative gate (`canUseTool`); read/analysis
tools are always allowed, document-mutating tools are gated.

## Layout

```
src/index.ts             API entrypoint: boots Fastify (REST + agent SSE + MCP)
src/worker.ts            worker entrypoint: boots the BullMQ analysis worker
src/http/                Fastify app + routes (auth, projects, jobs, billing, agent, mcp)
src/mcp/server.ts        true MCP server (Streamable HTTP, per-user/project auth)
src/jobs/                BullMQ connection / queue / worker
lib/agent/tool-defs.ts   the plot-hole toolset (single source of truth)
lib/agent/runner.ts      query() loop -> normalized AgentEvent stream
lib/agent/prompt.ts      system prompt (plot-hole + timeline focus)
lib/store/               Store interface + Drizzle (Postgres) + in-memory impls
lib/db/                  Drizzle schema + connection
test/                    store, auth, parsing tests (no network)
```

## Run

```bash
pnpm install
pnpm test && pnpm typecheck

# Full stack (Postgres + Redis + server + worker):
docker compose up --build          # from the repo's plot-hole-finder/ dir

# Or locally, with DATABASE_URL + REDIS_URL set:
pnpm db:migrate
ANTHROPIC_API_KEY=... pnpm start   # API server (port 3000)
ANTHROPIC_API_KEY=... pnpm worker  # background analysis worker
```

The Agent SDK runs in a Node subprocess; runtime is plain `tsx` (no build step).
