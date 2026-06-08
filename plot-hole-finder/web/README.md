# plot-hole-finder (unified TypeScript app)

One Next.js app: UI + API + the **Claude Agent SDK** loop + backend tools. This
replaces the Python backend — detection is the agent, so the value is the agent
loop, the tools, scale, ingestion, and (later) the 3-panel UI.

> Status: **server/agent core first; UI intentionally not built yet.**

## Architecture

The agent reads documents in slices and reasons about plot holes, continuity,
and **timeline/sequencing** — maintaining a retrieval-backed memory so it scales
past the context window. Capabilities are **tools** the agent calls:

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
app/api/agent/route.ts   SSE endpoint: runs the agent, streams normalized events
lib/agent/runner.ts      query() loop -> normalized AgentEvent stream
lib/agent/tools.ts       the backend tools (SDK in-process MCP)
lib/agent/permissions.ts ask/plan/auto/bypass gating
lib/agent/prompt.ts      system prompt (plot-hole + timeline focus)
lib/store/               Store interface + in-memory impl (Drizzle/Postgres next)
test/core.test.ts        store + permission gating tests (no network)
```

## Run

```bash
pnpm install
ANTHROPIC_API_KEY=... pnpm dev        # POST /api/agent  (SSE)
pnpm test && pnpm typecheck
```

The Agent SDK runs in a Node subprocess (route is `runtime = "nodejs"`).

## Next

- Drizzle/Postgres `Store` (durable, pgvector evidence) — replace in-memory.
- Port auth / billing / jobs / document parsing from the Python `server/`.
- Then the 3-panel UI (file tree · document editor w/ timeline notes · agent panel).
