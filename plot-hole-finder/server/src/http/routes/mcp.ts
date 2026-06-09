// Mounts the TRUE MCP server (Streamable HTTP) at /mcp. Auth + project scope are
// taken from the Authorization bearer token and the x-plothole-project header
// (or ?project=). The MCP SDK transport owns the raw response from here.

import type { FastifyInstance, FastifyRequest } from "fastify";
import { config } from "@/lib/config";
import { clientIp, enforceRateLimit } from "../context";
import { handleMcpRequest } from "../../mcp/server";

function scopeOf(req: FastifyRequest) {
  const projectId =
    (req.headers["x-plothole-project"] as string | undefined) ??
    (req.query as { project?: string } | undefined)?.project;
  return { authHeader: req.headers.authorization, projectId };
}

export async function mcpRoutes(app: FastifyInstance) {
  // Throttle by project (or client IP pre-auth) so MCP calls can't flood the
  // backend. Runs before hijack so a 429 is a normal JSON response.
  app.addHook("onRequest", async (req) => {
    if (!req.url.startsWith("/mcp")) return;
    const { projectId } = scopeOf(req);
    await enforceRateLimit("mcp", projectId ?? clientIp(req), config.mcpRateLimit);
  });

  app.post("/mcp", async (req, reply) => {
    reply.hijack();
    const { authHeader, projectId } = scopeOf(req);
    await handleMcpRequest(req.raw, reply.raw, { authHeader, projectId, body: req.body });
  });

  // GET (server-initiated SSE) and DELETE (session teardown) are part of the
  // Streamable HTTP spec; in stateless mode the transport replies appropriately.
  for (const method of ["GET", "DELETE"] as const) {
    app.route({
      method,
      url: "/mcp",
      handler: async (req, reply) => {
        reply.hijack();
        const { authHeader, projectId } = scopeOf(req);
        await handleMcpRequest(req.raw, reply.raw, { authHeader, projectId, body: undefined });
      },
    });
  }
}
