// Interactive agent route: runs the Claude Agent SDK rolling-memory loop over a
// project's document (durable store) and streams normalized events to the UI as
// Server-Sent Events. Abortable when the client disconnects.
//
// This is the live, interactive path (the UI's agent panel). Long unattended
// passes over massive documents go through the BullMQ queue instead
// (POST /api/projects/:id/analyze|fix).

import type { FastifyInstance } from "fastify";
import { config } from "@/lib/config";
import { getProject } from "@/lib/projects";
import { DrizzleStore } from "@/lib/store/drizzle";
import { runAgent } from "@/lib/agent/runner";
import type { UiPermissionMode } from "@/lib/agent/permissions";
import { enforceRateLimit, requireUser, sendError } from "../context";

const MODES: UiPermissionMode[] = ["ask", "plan", "auto", "bypass"];

interface AgentBody {
  projectId: string;
  prompt?: string;
  model?: string;
  mode?: UiPermissionMode;
}

export async function agentRoutes(app: FastifyInstance) {
  app.post("/api/agent", async (req, reply) => {
    let user;
    try {
      user = await requireUser(req);
      await enforceRateLimit("agent", user.id, config.agentRateLimit);
    } catch (err) {
      return sendError(reply, err);
    }

    const b = (req.body ?? {}) as Partial<AgentBody>;
    const projectId = b.projectId ?? "";
    const prompt = b.prompt ?? "Analyze this document for plot holes and timeline/sequencing problems.";
    const model = b.model ?? config.anthropicModel;
    const mode: UiPermissionMode = b.mode && MODES.includes(b.mode) ? b.mode : "ask";

    try {
      await getProject(user.id, projectId); // ownership / existence
    } catch (err) {
      return sendError(reply, err);
    }

    const store = new DrizzleStore();
    const ac = new AbortController();
    req.raw.on("close", () => ac.abort());

    reply.raw.writeHead(200, {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
    });
    reply.hijack();

    const send = (event: unknown) => reply.raw.write(`data: ${JSON.stringify(event)}\n\n`);

    try {
      for await (const ev of runAgent({
        store, projectId, prompt, mode, model, signal: ac.signal,
        maxTurns: config.agentMaxTurns, maxBudgetUsd: config.agentMaxBudgetUsd,
      })) {
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
      reply.raw.write("event: done\ndata: {}\n\n");
      reply.raw.end();
    }
  });
}
