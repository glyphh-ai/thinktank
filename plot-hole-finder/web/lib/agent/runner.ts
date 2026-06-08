// Runs the Claude Agent SDK loop with the backend plot-hole tools and yields a
// normalized event stream the UI can render Claude-style (text, thinking,
// tool-use, tool-result, result).

import { createSdkMcpServer, query } from "@anthropic-ai/claude-agent-sdk";
import { SYSTEM_PROMPT } from "./prompt";
import { buildCanUseTool, toSdkPermissionMode, type Approver, type UiPermissionMode } from "./permissions";
import { buildTools, MCP_SERVER_NAME } from "./tools";
import type { Store } from "../store/types";

export type AgentEvent =
  | { type: "system"; subtype: string }
  | { type: "text"; text: string }
  | { type: "thinking"; text: string }
  | { type: "tool_use"; id: string; name: string; input: unknown }
  | { type: "tool_result"; id?: string; text: string }
  | { type: "result"; subtype: string; costUsd?: number }
  | { type: "error"; message: string };

export interface RunAgentArgs {
  store: Store;
  projectId: string;
  prompt: string;
  mode: UiPermissionMode;
  model?: string;
  approver?: Approver;
  maxTurns?: number;
  maxBudgetUsd?: number; // hard cost cap (hardening)
  signal?: AbortSignal; // cancel/interrupt the run
}

function shortToolName(name: string): string {
  return name.startsWith(`mcp__${MCP_SERVER_NAME}__`) ? name.split("__").slice(2).join("__") : name;
}

export async function* runAgent(args: RunAgentArgs): AsyncGenerator<AgentEvent> {
  const { store, projectId, prompt, mode, model, approver, maxTurns, maxBudgetUsd, signal } = args;
  const server = createSdkMcpServer({ name: MCP_SERVER_NAME, tools: buildTools(store, projectId) });

  let q: { interrupt(): Promise<void> } & AsyncGenerator<unknown>;
  try {
    q = query({
      prompt,
      options: {
        model,
        systemPrompt: SYSTEM_PROMPT,
        mcpServers: { [MCP_SERVER_NAME]: { type: "sdk", name: MCP_SERVER_NAME, instance: server.instance } },
        permissionMode: toSdkPermissionMode(mode),
        canUseTool: buildCanUseTool(mode, approver),
        maxTurns: maxTurns ?? 80,
        ...(maxBudgetUsd !== undefined ? { maxBudgetUsd } : {}),
      },
    }) as unknown as { interrupt(): Promise<void> } & AsyncGenerator<unknown>;
  } catch (err) {
    yield { type: "error", message: err instanceof Error ? err.message : String(err) };
    return;
  }

  if (signal) signal.addEventListener("abort", () => void q.interrupt().catch(() => {}), { once: true });
  const stream = q;

  try {
    for await (const msg of stream) {
      const m = msg as any;
      switch (m.type) {
        case "system":
          yield { type: "system", subtype: m.subtype ?? "init" };
          break;
        case "assistant":
          for (const block of m.message?.content ?? []) {
            if (block.type === "text" && block.text) yield { type: "text", text: block.text };
            else if (block.type === "thinking" && block.thinking) yield { type: "thinking", text: block.thinking };
            else if (block.type === "tool_use")
              yield { type: "tool_use", id: block.id, name: shortToolName(block.name), input: block.input };
          }
          break;
        case "user":
          for (const block of m.message?.content ?? []) {
            if (block.type === "tool_result") {
              const text = Array.isArray(block.content)
                ? block.content.map((c: any) => c.text ?? "").join("")
                : String(block.content ?? "");
              yield { type: "tool_result", id: block.tool_use_id, text };
            }
          }
          break;
        case "result":
          yield { type: "result", subtype: m.subtype ?? "success", costUsd: m.total_cost_usd };
          break;
      }
    }
  } catch (err) {
    yield { type: "error", message: err instanceof Error ? err.message : String(err) };
  }
}
