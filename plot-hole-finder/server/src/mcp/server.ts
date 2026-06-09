// TRUE MCP server (@modelcontextprotocol/sdk) exposed over Streamable HTTP.
//
// This is a real, standalone Model Context Protocol endpoint — not in-process
// Agent SDK tools. Any MCP client (Claude Desktop, the Agent SDK as an external
// client, etc.) can connect to POST /mcp and call the plot-hole toolset.
//
// It is per-user/project authenticated: the caller presents a Bearer access
// token (same JWT as the REST API) and selects a project via the
// `x-plothole-project` header (or `?project=` query). Tools then operate ONLY on
// that user's project, backed by the durable Drizzle store. Stateless mode: a
// fresh McpServer + transport is built per request so auth/scope are re-checked
// every call and the endpoint scales horizontally.

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import type { IncomingMessage, ServerResponse } from "node:http";
import { z } from "zod";
import { MCP_SERVER_NAME, plotHoleTools } from "@/lib/agent/tool-defs";
import { userFromAuthHeader } from "@/lib/auth/service";
import { config } from "@/lib/config";
import { getProject } from "@/lib/projects";
import { DrizzleStore } from "@/lib/store/drizzle";
import { AppError, Forbidden, Unauthorized, ValidationError } from "@/lib/errors";

export interface McpScope {
  userId: string;
  projectId: string;
}

/**
 * DNS-rebinding protection: reject Host headers not in the allowlist (the SDK's
 * built-in option is deprecated in favor of external middleware like this).
 * Empty allowlist disables the check (dev).
 */
export function assertAllowedHost(host: string | undefined): void {
  if (config.mcpAllowedHosts.length === 0) return;
  const bare = (host ?? "").split(":")[0]!.toLowerCase();
  const ok = config.mcpAllowedHosts.some((h) => bare === h.split(":")[0]!.toLowerCase());
  if (!ok) throw new Forbidden("host not allowed");
}

/** Authenticate the request and resolve the project scope, or throw AppError. */
export async function resolveScope(
  authHeader: string | undefined,
  projectId: string | undefined,
): Promise<McpScope> {
  const user = await userFromAuthHeader(authHeader ?? null);
  if (!projectId) {
    throw new ValidationError("project is required (set the x-plothole-project header or ?project=)");
  }
  await getProject(user.id, projectId); // ownership / existence check
  return { userId: user.id, projectId };
}

/** Build a fresh MCP server scoped to one user's project. */
export function buildMcpServer(scope: McpScope): McpServer {
  const server = new McpServer(
    { name: MCP_SERVER_NAME, version: "0.1.0" },
    { capabilities: { tools: {} } },
  );
  const store = new DrizzleStore();
  const ctx = { store, projectId: scope.projectId };

  for (const t of plotHoleTools) {
    server.registerTool(
      t.name,
      {
        description: t.description,
        inputSchema: t.schema as z.ZodRawShape,
        annotations: { readOnlyHint: !t.mutating },
      },
      async (args: Record<string, unknown>) => {
        const text = await t.run(ctx, args as never);
        return { content: [{ type: "text" as const, text }] };
      },
    );
  }

  return server;
}

/**
 * Handle one MCP HTTP request end-to-end (stateless). Designed to be called from
 * a Fastify route with the raw Node req/res and the already-parsed JSON body.
 */
export async function handleMcpRequest(
  req: IncomingMessage,
  res: ServerResponse,
  opts: { authHeader?: string; projectId?: string; body: unknown },
): Promise<void> {
  let scope: McpScope;
  try {
    assertAllowedHost(req.headers.host);
    scope = await resolveScope(opts.authHeader, opts.projectId);
  } catch (err) {
    const e = err instanceof AppError ? err : new Unauthorized("unauthorized");
    res.writeHead(e.statusCode, { "content-type": "application/json" });
    res.end(JSON.stringify({
      jsonrpc: "2.0",
      error: { code: -32001, message: e.message },
      id: null,
    }));
    return;
  }

  const server = buildMcpServer(scope);
  // Stateless: no session id, JSON responses (no long-lived SSE per call).
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: undefined,
    enableJsonResponse: true,
  });

  res.on("close", () => {
    void transport.close();
    void server.close();
  });

  await server.connect(transport);
  await transport.handleRequest(req, res, opts.body);
}
