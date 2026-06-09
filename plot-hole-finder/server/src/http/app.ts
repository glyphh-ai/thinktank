// Builds the Fastify application: control-plane REST API (auth, projects, jobs,
// billing), the interactive agent SSE endpoint, and the mounted TRUE MCP server.

import Fastify, { type FastifyInstance } from "fastify";
import cors from "@fastify/cors";
import multipart from "@fastify/multipart";
import { AppError } from "@/lib/errors";
import { closeDb, pingDb } from "@/lib/db";
import { closeRedis, getRedis } from "@/lib/redis";
import { authRoutes } from "./routes/auth";
import { projectRoutes } from "./routes/projects";
import { jobRoutes } from "./routes/jobs";
import { billingRoutes } from "./routes/billing";
import { agentRoutes } from "./routes/agent";
import { mcpRoutes } from "./routes/mcp";
import { closeQueue } from "../jobs/queue";

export async function buildApp(opts: { logger?: boolean } = {}): Promise<FastifyInstance> {
  const app = Fastify({ logger: opts.logger ?? true, bodyLimit: 50 * 1024 * 1024 });

  // JSON parser that also retains the raw bytes (needed for Stripe webhook
  // signature verification).
  app.addContentTypeParser("application/json", { parseAs: "buffer" }, (req, body, done) => {
    (req as { rawBody?: Buffer }).rawBody = body as Buffer;
    try {
      const buf = body as Buffer;
      done(null, buf.length ? JSON.parse(buf.toString("utf8")) : {});
    } catch (err) {
      done(err as Error);
    }
  });

  await app.register(cors, { origin: true });
  await app.register(multipart, { limits: { fileSize: 50 * 1024 * 1024 } });

  // Map domain errors to clean JSON; log the rest as 500s.
  app.setErrorHandler((err, _req, reply) => {
    if (err instanceof AppError) {
      return reply.code(err.statusCode).send({ error: err.code, message: err.message });
    }
    // Fastify validation / payload errors carry a statusCode.
    if (typeof (err as { statusCode?: number }).statusCode === "number") {
      const code = (err as { statusCode: number }).statusCode;
      return reply.code(code).send({ error: "request_error", message: (err as Error).message });
    }
    reply.log.error(err);
    return reply.code(500).send({ error: "internal_error", message: "internal error" });
  });

  // Liveness: process is up. Readiness: dependencies (Postgres + Redis) reachable.
  app.get("/health", async () => ({ ok: true }));
  app.get("/ready", async (_req, reply) => {
    const checks: Record<string, "ok" | string> = {};
    try { await pingDb(); checks.db = "ok"; }
    catch (err) { checks.db = err instanceof Error ? err.message : "error"; }
    const redis = getRedis();
    if (!redis) checks.redis = "not configured";
    else {
      try { await redis.ping(); checks.redis = "ok"; }
      catch (err) { checks.redis = err instanceof Error ? err.message : "error"; }
    }
    const ready = checks.db === "ok" && (checks.redis === "ok" || checks.redis === "not configured");
    return reply.code(ready ? 200 : 503).send({ ready, checks });
  });

  // Close shared connections (Postgres pool, caches, BullMQ queue) on shutdown.
  app.addHook("onClose", async () => {
    await Promise.allSettled([closeQueue(), closeRedis(), closeDb()]);
  });

  await app.register(authRoutes);
  await app.register(projectRoutes);
  await app.register(jobRoutes);
  await app.register(billingRoutes);
  await app.register(agentRoutes);
  await app.register(mcpRoutes);

  return app;
}
