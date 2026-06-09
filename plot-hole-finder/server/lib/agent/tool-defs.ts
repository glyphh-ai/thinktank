// Single source of truth for the plot-hole toolset.
//
// Each tool is declared once here as a pure definition (name + zod input shape
// + a handler that operates over a Store/projectId). Two adapters consume these:
//   - lib/agent/tools.ts wraps them as Claude Agent SDK in-process `tool()`s
//     (used by the worker's runAgent loop).
//   - src/mcp/server.ts registers them on a TRUE @modelcontextprotocol/sdk
//     McpServer exposed over Streamable HTTP (per-user/project authenticated).
// Keeping the definitions here means the agent loop and the external MCP server
// can never drift apart.

import { z } from "zod";
import type { Store } from "../store/types";

export const MCP_SERVER_NAME = "plothole";

export interface ToolContext {
  store: Store;
  projectId: string;
}

export interface PlotHoleTool<Shape extends z.ZodRawShape = z.ZodRawShape> {
  name: string;
  description: string;
  schema: Shape;
  /** Mutates the user's document — gated by permission mode / MCP scope. */
  mutating?: boolean;
  run: (ctx: ToolContext, args: z.infer<z.ZodObject<Shape>>) => Promise<string>;
}

// Helper so handlers can be written generically.
function def<Shape extends z.ZodRawShape>(t: PlotHoleTool<Shape>): PlotHoleTool {
  return t as unknown as PlotHoleTool;
}

export const plotHoleTools: PlotHoleTool[] = [
  def({
    name: "read_document",
    description: "Read a slice of the document. Advance `start` to page through large documents.",
    schema: {
      start: z.number().int().min(0).default(0),
      length: z.number().int().min(1).max(40000).default(12000),
    },
    async run({ store, projectId }, { start, length }) {
      const doc = await store.getDocument(projectId);
      const slice = doc.slice(start, start + length);
      return JSON.stringify({
        total_chars: doc.length,
        start,
        end: start + slice.length,
        more: start + slice.length < doc.length,
        text: slice,
      });
    },
  }),

  def({
    name: "recall_memory",
    description: "Recall durable facts already established about the given entities (scales past context).",
    schema: { entities: z.array(z.string()).min(1) },
    async run({ store, projectId }, { entities }) {
      return JSON.stringify(await store.getMemory(projectId, entities));
    },
  }),

  def({
    name: "remember_facts",
    description: "Persist durable facts/states/events to memory for later cross-checking.",
    schema: {
      facts: z.array(z.object({
        entity: z.string(),
        key: z.string(),
        value: z.string(),
        kind: z.enum(["state", "event"]).default("state"),
        chunk_index: z.number().int().default(0),
        story_time: z.number().int().nullable().optional(),
        char_start: z.number().int().optional(),
      })).min(1),
    },
    async run({ store, projectId }, { facts }) {
      await store.addMemory(projectId, facts.map((f) => ({
        entity: f.entity, key: f.key, value: f.value, kind: f.kind,
        chunkIndex: f.chunk_index, storyTime: f.story_time ?? null, charStart: f.char_start,
      })));
      return `remembered ${facts.length} fact(s)`;
    },
  }),

  def({
    name: "record_issue",
    description: "Record a plot hole / inconsistency with evidence spans.",
    schema: {
      category: z.enum(["continuity", "timeline", "character", "knowledge", "broken_reference", "sequencing"]),
      severity: z.enum(["low", "medium", "high"]).default("medium"),
      title: z.string(),
      description: z.string(),
      evidence: z.array(z.object({
        role: z.enum(["state_a", "state_b", "context"]).default("context"),
        char_start: z.number().int(),
        char_end: z.number().int(),
        quote: z.string(),
      })).default([]),
      suggested_fix: z.string().optional(),
    },
    async run({ store, projectId }, { category, severity, title, description, evidence, suggested_fix }) {
      const issue = await store.addIssue(projectId, {
        category, severity, title, description, suggestedFix: suggested_fix,
        evidence: evidence.map((e) => ({ role: e.role, charStart: e.char_start, charEnd: e.char_end, quote: e.quote })),
      });
      return `recorded issue ${issue.id}: ${title}`;
    },
  }),

  def({
    name: "record_timeline_event",
    description: "Record an event for the timeline with narrative_time (position) and story_time (in-world order).",
    schema: {
      summary: z.string(),
      entity: z.string().optional(),
      narrative_time: z.number().int(),
      story_time: z.number().int(),
      char_start: z.number().int(),
      char_end: z.number().int(),
    },
    async run({ store, projectId }, { summary, entity, narrative_time, story_time, char_start, char_end }) {
      const ev = await store.addTimelineEvent(projectId, {
        summary, entity, narrativeTime: narrative_time, storyTime: story_time,
        charStart: char_start, charEnd: char_end,
      });
      return `recorded timeline event ${ev.id}`;
    },
  }),

  def({
    name: "get_timeline",
    description: "Get the reconstructed timeline so far (to check sequencing).",
    schema: {},
    async run({ store, projectId }) {
      return JSON.stringify(await store.getTimeline(projectId));
    },
  }),

  def({
    name: "add_note",
    description: "Flag a span on the document timeline for the author.",
    schema: {
      char_start: z.number().int(),
      char_end: z.number().int(),
      text: z.string(),
      kind: z.enum(["note", "plot_hole", "sequencing"]).default("note"),
      issue_id: z.string().optional(),
    },
    async run({ store, projectId }, { char_start, char_end, text, kind, issue_id }) {
      const note = await store.addNote(projectId, {
        charStart: char_start, charEnd: char_end, text, kind, issueId: issue_id,
      });
      return `added note ${note.id}`;
    },
  }),

  def({
    name: "list_issues",
    description: "List all recorded issues.",
    schema: {},
    async run({ store, projectId }) {
      return JSON.stringify(await store.listIssues(projectId));
    },
  }),

  def({
    name: "apply_edit",
    description: "Apply an edit to the document (replace a char range). Gated by permission mode.",
    mutating: true,
    schema: {
      char_start: z.number().int(),
      char_end: z.number().int(),
      replacement: z.string(),
    },
    async run({ store, projectId }, { char_start, char_end, replacement }) {
      await store.applyEdit(projectId, char_start, char_end, replacement);
      return `applied edit [${char_start},${char_end}]`;
    },
  }),

  def({
    name: "export_document",
    description: "Return the current document text for saving/export.",
    schema: {},
    async run({ store, projectId }) {
      return store.getDocument(projectId);
    },
  }),
];

// Fully-qualified names of mutating tools, as the Agent SDK exposes them
// (mcp__<server>__<tool>). Used by the permission gate.
export const MUTATING_TOOLS = new Set(
  plotHoleTools.filter((t) => t.mutating).map((t) => `mcp__${MCP_SERVER_NAME}__${t.name}`),
);
