import { describe, expect, it } from "vitest";
import { InMemoryStore } from "../lib/store/memory";
import { buildCanUseTool } from "../lib/agent/permissions";
import { MUTATING_TOOLS } from "../lib/agent/tools";

const EDIT = "mcp__plothole__apply_edit";
const READ = "mcp__plothole__read_document";

describe("InMemoryStore", () => {
  it("round-trips document, memory, issues, timeline, notes", async () => {
    const s = new InMemoryStore({ id: "p", documentText: "Hello world." });
    expect(await s.getDocument("p")).toBe("Hello world.");

    await s.applyEdit("p", 0, 5, "Hi");
    expect(await s.getDocument("p")).toBe("Hi world.");

    await s.addMemory("p", [
      { entity: "Elena", key: "eye_color", value: "green", kind: "state", chunkIndex: 0 },
      { entity: "Marcus", key: "hair", value: "black", kind: "state", chunkIndex: 0 },
    ]);
    const recalled = await s.getMemory("p", ["elena"]); // case-insensitive
    expect(recalled).toHaveLength(1);
    expect(recalled[0]!.value).toBe("green");

    const issue = await s.addIssue("p", {
      category: "continuity", severity: "high", title: "Eye colour", description: "green->blue",
      evidence: [{ role: "state_a", charStart: 0, charEnd: 5, quote: "green" }],
    });
    expect((await s.listIssues("p"))[0]!.id).toBe(issue.id);

    await s.addTimelineEvent("p", {
      summary: "arrival", narrativeTime: 0, storyTime: 0, charStart: 0, charEnd: 5,
    });
    expect(await s.getTimeline("p")).toHaveLength(1);

    await s.addNote("p", { charStart: 0, charEnd: 5, text: "check this", kind: "plot_hole" });
    expect(await s.listNotes("p")).toHaveLength(1);
  });
});

describe("permission gating", () => {
  const input = { char_start: 0, char_end: 1, replacement: "x" };

  it("always allows read/analysis tools", async () => {
    for (const mode of ["ask", "plan", "auto", "bypass"] as const) {
      const can = buildCanUseTool(mode);
      expect((await can(READ, {})).behavior).toBe("allow");
    }
  });

  it("gates document edits by mode", async () => {
    expect((await buildCanUseTool("auto")(EDIT, input)).behavior).toBe("allow");
    expect((await buildCanUseTool("bypass")(EDIT, input)).behavior).toBe("allow");
    expect((await buildCanUseTool("plan")(EDIT, input)).behavior).toBe("deny");
    expect((await buildCanUseTool("ask")(EDIT, input)).behavior).toBe("deny"); // no approver
    expect((await buildCanUseTool("ask", async () => true)(EDIT, input)).behavior).toBe("allow");
  });

  it("knows which tools mutate the document", () => {
    expect(MUTATING_TOOLS.has(EDIT)).toBe(true);
    expect(MUTATING_TOOLS.has(READ)).toBe(false);
  });
});
