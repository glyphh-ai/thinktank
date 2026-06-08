// Interactive agent endpoint: runs the Claude Agent SDK loop over a project's
// document (durable store) and streams normalized events as SSE. Abortable.

import { requireUser } from "@/lib/http";
import { AppError } from "@/lib/errors";
import { getProject } from "@/lib/projects";
import { DrizzleStore } from "@/lib/store/drizzle";
import { runAgent } from "@/lib/agent/runner";
import type { UiPermissionMode } from "@/lib/agent/permissions";
import { config } from "@/lib/config";

export const runtime = "nodejs";

const MODES: UiPermissionMode[] = ["ask", "plan", "auto", "bypass"];

export async function POST(req: Request) {
  let user: { id: string; email: string };
  try {
    user = await requireUser(req);
  } catch (err) {
    const e = err as AppError;
    return Response.json({ error: e.code ?? "unauthorized", message: e.message }, { status: e.statusCode ?? 401 });
  }

  const b = await req.json().catch(() => ({}));
  const projectId: string = b.projectId;
  const prompt: string = b.prompt ?? "Analyze this document for plot holes and timeline/sequencing problems.";
  const model: string = b.model ?? config.anthropicModel;
  const mode: UiPermissionMode = MODES.includes(b.mode) ? b.mode : "ask";

  try {
    await getProject(user.id, projectId); // ownership / existence
  } catch (err) {
    const e = err as AppError;
    return Response.json({ error: e.code ?? "not_found", message: e.message }, { status: e.statusCode ?? 404 });
  }

  const store = new DrizzleStore();
  const encoder = new TextEncoder();
  const ac = new AbortController();
  req.signal.addEventListener("abort", () => ac.abort());

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (event: unknown) => controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
      try {
        for await (const ev of runAgent({ store, projectId, prompt, mode, model, signal: ac.signal })) {
          send(ev);
        }
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
    cancel() { ac.abort(); },
  });

  return new Response(stream, {
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
    },
  });
}
