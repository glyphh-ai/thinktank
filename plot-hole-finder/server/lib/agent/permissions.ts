// UI permission modes -> Claude Agent SDK permissionMode + a canUseTool gate.
// Read/analysis tools are always allowed; document-mutating tools are gated.

import { MUTATING_TOOLS } from "./tools";

export type UiPermissionMode = "ask" | "plan" | "auto" | "bypass";

// Asked at edit time in "ask" mode (wired to the UI's approve/deny later).
export type Approver = (toolName: string, input: Record<string, unknown>) => Promise<boolean>;

export type PermissionResult =
  | { behavior: "allow"; updatedInput: Record<string, unknown> }
  | { behavior: "deny"; message: string };

export function toSdkPermissionMode(mode: UiPermissionMode): "default" | "plan" {
  // canUseTool is our authoritative gate; keep the SDK in 'default' so it's always
  // consulted, except 'plan' which also stops the model from attempting changes.
  return mode === "plan" ? "plan" : "default";
}

export function buildCanUseTool(mode: UiPermissionMode, approver?: Approver) {
  return async (
    toolName: string,
    input: Record<string, unknown>,
  ): Promise<PermissionResult> => {
    if (!MUTATING_TOOLS.has(toolName)) {
      return { behavior: "allow", updatedInput: input }; // read / analysis
    }
    switch (mode) {
      case "bypass":
      case "auto":
        return { behavior: "allow", updatedInput: input };
      case "plan":
        return { behavior: "deny", message: "Plan mode: no document changes." };
      case "ask":
        if (approver && (await approver(toolName, input))) {
          return { behavior: "allow", updatedInput: input };
        }
        return { behavior: "deny", message: "Edit requires the author's approval." };
    }
  };
}
