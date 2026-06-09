// Project routes: create from text or upload, list/get, read issues/timeline,
// and enqueue/cancel analyze/fix jobs on the BullMQ queue.

import type { FastifyInstance } from "fastify";
import { z } from "zod";
import { currentPlan } from "@/lib/billing";
import { Forbidden, NotFound, ValidationError } from "@/lib/errors";
import { limitsFor } from "@/lib/plans";
import { createProject, getProject, listProjects } from "@/lib/projects";
import { DrizzleStore } from "@/lib/store/drizzle";
import { cancelJob, enqueueAnalysis } from "../../jobs/queue";
import { parseBody, requireUser } from "../context";

const createSchema = z.object({ name: z.string().max(200).default(""), text: z.string().min(1) });

export async function projectRoutes(app: FastifyInstance) {
  app.get("/api/projects", async (req) => {
    const user = await requireUser(req);
    return listProjects(user.id);
  });

  app.post("/api/projects", async (req, reply) => {
    const user = await requireUser(req);
    const { name, text } = parseBody(req, createSchema);
    return reply.code(201).send(await createProject(user.id, name, { text }));
  });

  app.post("/api/projects/upload", async (req, reply) => {
    const user = await requireUser(req);
    const file = await req.file();
    if (!file) throw new ValidationError("file is required");
    const name = file.fields?.name && "value" in (file.fields.name as { value?: unknown })
      ? String((file.fields.name as { value: unknown }).value)
      : file.filename;
    const data = new Uint8Array(await file.toBuffer());
    return reply.code(201).send(await createProject(user.id, name, { filename: file.filename, data }));
  });

  app.get("/api/projects/:id", async (req) => {
    const user = await requireUser(req);
    const { id } = req.params as { id: string };
    return getProject(user.id, id);
  });

  app.get("/api/projects/:id/issues", async (req) => {
    const user = await requireUser(req);
    const { id } = req.params as { id: string };
    await getProject(user.id, id);
    return new DrizzleStore().listIssues(id);
  });

  app.get("/api/projects/:id/timeline", async (req) => {
    const user = await requireUser(req);
    const { id } = req.params as { id: string };
    await getProject(user.id, id);
    const store = new DrizzleStore();
    return { timeline: await store.getTimeline(id), notes: await store.listNotes(id) };
  });

  app.post("/api/projects/:id/analyze", async (req, reply) => {
    const user = await requireUser(req);
    const { id } = req.params as { id: string };
    await getProject(user.id, id);
    const { jobId, state, deduped } = await enqueueAnalysis({ type: "analyze", userId: user.id, projectId: id });
    return reply.code(202).send({ jobId, status: state, deduped });
  });

  app.post("/api/projects/:id/fix", async (req, reply) => {
    const user = await requireUser(req);
    const { id } = req.params as { id: string };
    await getProject(user.id, id);
    if (!limitsFor(await currentPlan(user.id)).canFix) {
      throw new Forbidden("fix-this-for-me requires the Pro plan");
    }
    const { jobId, state, deduped } = await enqueueAnalysis({ type: "fix", userId: user.id, projectId: id });
    return reply.code(202).send({ jobId, status: state, deduped });
  });

  // Cancel an in-flight or queued job for one of the caller's projects.
  app.post("/api/jobs/:id/cancel", async (req) => {
    const user = await requireUser(req);
    const { id } = req.params as { id: string };
    const res = await cancelJob(id);
    if (!res || res.userId !== user.id) throw new NotFound("job not found");
    return { jobId: id, status: "cancelling", was: res.state };
  });
}
