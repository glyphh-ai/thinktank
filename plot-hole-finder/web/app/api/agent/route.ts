// Server endpoint: runs the agent loop for a request and streams normalized
// events as SSE. (No-DB for now: seeds an in-memory store from the request; the
// Drizzle/Postgres store is the next step. UI is intentionally not built yet.)

import { runAgent } from "@/lib/agent/runner";
import type { UiPermissionMode } from "@/lib/agent/permissions";
import { InMemoryStore } from "@/lib/store/memory";

export const runtime = "nodejs"; // the Agent SDK spawns a Node subprocess

const MODES: UiPermissionMode[] = ["ask", "plan", "auto", "bypass"];

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}));
  const projectId: string = body.projectId ?? "demo";
  const documentText: string = body.documentText ?? "";
  const prompt: string = body.prompt ?? "Analyze this document for plot holes and timeline/sequencing problems.";
  const model: string | undefined = body.model;
  const mode: UiPermissionMode = MODES.includes(body.mode) ? body.mode : "ask";

  if (!documentText.trim()) {
    return new Response(JSON.stringify({ error: "documentText is required" }), {
      status: 400, headers: { "content-type": "application/json" },
    });
  }

  const store = new InMemoryStore({ id: projectId, documentText });
  const encoder = new TextEncoder();

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (event: unknown) =>
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
      try {
        for await (const ev of runAgent({ store, projectId, prompt, mode, model })) {
          send(ev);
        }
        // Final snapshot so the client can render issues/timeline/notes/doc.
        send({
          type: "snapshot",
          issues: await store.listIssues(projectId),
          timeline: await store.getTimeline(projectId),
          notes: await store.listNotes(projectId),
          document: await store.getDocument(projectId),
        });
      } catch (err) {
        send({ type: "error", message: err instanceof Error ? err.message : String(err) });
      } finally {
        controller.enqueue(encoder.encode("event: done\ndata: {}\n\n"));
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
    },
  });
}
