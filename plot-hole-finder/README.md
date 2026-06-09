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

One **TypeScript Next.js app** in [`web/`](web/): the UI + API + a **Claude Agent
SDK** loop whose backend **tools** read the document, maintain a retrieval-backed
memory (so it scales past the context window), and record issues / timeline /
notes / edits. Permission modes (ask / plan / auto / bypass) gate document edits.

Detection is Claude's — established by the evals in [`web/`](web/) and recorded in
the project history: the agent + memory approach beat a bespoke extraction/HDC
pipeline, and at >1M tokens it's the only thing that runs. So the product value is
the agent loop, the tools, scale, ingestion, and the (in-progress) UI.

See [`web/README.md`](web/README.md) to run it.

> `DESIGN.md` documents the original (HDC/glyphh) design and is kept as historical
> record; that approach was benchmarked out and removed. The Python prototype has
> been deleted in favor of the unified TypeScript app.
