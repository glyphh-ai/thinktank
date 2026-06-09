// Adapter: wraps the shared plot-hole tool definitions (lib/agent/tool-defs.ts)
// as Claude Agent SDK in-process MCP tools for the runAgent loop. The same
// definitions are also served by the TRUE MCP server (src/mcp/server.ts).

import { tool } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";
import type { Store } from "../store/types";
import { MCP_SERVER_NAME, MUTATING_TOOLS, plotHoleTools } from "./tool-defs";

export { MCP_SERVER_NAME, MUTATING_TOOLS };

const ok = (text: string) => ({ content: [{ type: "text" as const, text }] });

export function buildTools(store: Store, projectId: string) {
  return plotHoleTools.map((t) =>
    tool(
      t.name,
      t.description,
      t.schema as z.ZodRawShape,
      async (args: Record<string, unknown>) => ok(await t.run({ store, projectId }, args as never)),
    ),
  );
}
