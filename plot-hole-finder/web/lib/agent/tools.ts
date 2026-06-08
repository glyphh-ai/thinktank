// Backend tools the agent calls (Claude Agent SDK in-process MCP tools).
// Built per-request so each closes over the Store + projectId.

import { tool } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";
import type { Store } from "../store/types";

export const MCP_SERVER_NAME = "plothole";

// Tools that change the user's document — gated by permission mode.
export const MUTATING_TOOLS = new Set([`mcp__${MCP_SERVER_NAME}__apply_edit`]);

const ok = (text: string) => ({ content: [{ type: "text" as const, text }] });

export function buildTools(store: Store, projectId: string) {
  return [
    tool(
      "read_document",
      "Read a slice of the document. Advance `start` to page through large documents.",
      { start: z.number().int().min(0).default(0), length: z.number().int().min(1).max(40000).default(12000) },
      async ({ start, length }) => {
        const doc = await store.getDocument(projectId);
        const slice = doc.slice(start, start + length);
        return ok(JSON.stringify({
          total_chars: doc.length, start, end: start + slice.length,
          more: start + slice.length < doc.length, text: slice,
        }));
      },
    ),

    tool(
      "recall_memory",
      "Recall durable facts already established about the given entities (scales past context).",
      { entities: z.array(z.string()).min(1) },
      async ({ entities }) => ok(JSON.stringify(await store.getMemory(projectId, entities))),
    ),

    tool(
      "remember_facts",
      "Persist durable facts/states/events to memory for later cross-checking.",
      {
        facts: z.array(z.object({
          entity: z.string(), key: z.string(), value: z.string(),
          kind: z.enum(["state", "event"]).default("state"),
          chunk_index: z.number().int().default(0),
          story_time: z.number().int().nullable().optional(),
          char_start: z.number().int().optional(),
        })).min(1),
      },
      async ({ facts }) => {
        await store.addMemory(projectId, facts.map((f) => ({
          entity: f.entity, key: f.key, value: f.value, kind: f.kind,
          chunkIndex: f.chunk_index, storyTime: f.story_time ?? null, charStart: f.char_start,
        })));
        return ok(`remembered ${facts.length} fact(s)`);
      },
    ),

    tool(
      "record_issue",
      "Record a plot hole / inconsistency with evidence spans.",
      {
        category: z.enum(["continuity", "timeline", "character", "knowledge", "broken_reference", "sequencing"]),
        severity: z.enum(["low", "medium", "high"]).default("medium"),
        title: z.string(),
        description: z.string(),
        evidence: z.array(z.object({
          role: z.enum(["state_a", "state_b", "context"]).default("context"),
          char_start: z.number().int(), char_end: z.number().int(), quote: z.string(),
        })).default([]),
        suggested_fix: z.string().optional(),
      },
      async ({ category, severity, title, description, evidence, suggested_fix }) => {
        const issue = await store.addIssue(projectId, {
          category, severity, title, description, suggestedFix: suggested_fix,
          evidence: evidence.map((e) => ({ role: e.role, charStart: e.char_start, charEnd: e.char_end, quote: e.quote })),
        });
        return ok(`recorded issue ${issue.id}: ${title}`);
      },
    ),

    tool(
      "record_timeline_event",
      "Record an event for the timeline with narrative_time (position) and story_time (in-world order).",
      {
        summary: z.string(), entity: z.string().optional(),
        narrative_time: z.number().int(), story_time: z.number().int(),
        char_start: z.number().int(), char_end: z.number().int(),
      },
      async ({ summary, entity, narrative_time, story_time, char_start, char_end }) => {
        const ev = await store.addTimelineEvent(projectId, {
          summary, entity, narrativeTime: narrative_time, storyTime: story_time,
          charStart: char_start, charEnd: char_end,
        });
        return ok(`recorded timeline event ${ev.id}`);
      },
    ),

    tool(
      "get_timeline",
      "Get the reconstructed timeline so far (to check sequencing).",
      {},
      async () => ok(JSON.stringify(await store.getTimeline(projectId))),
    ),

    tool(
      "add_note",
      "Flag a span on the document timeline for the author.",
      {
        char_start: z.number().int(), char_end: z.number().int(), text: z.string(),
        kind: z.enum(["note", "plot_hole", "sequencing"]).default("note"),
        issue_id: z.string().optional(),
      },
      async ({ char_start, char_end, text, kind, issue_id }) => {
        const note = await store.addNote(projectId, {
          charStart: char_start, charEnd: char_end, text, kind, issueId: issue_id,
        });
        return ok(`added note ${note.id}`);
      },
    ),

    tool(
      "list_issues",
      "List all recorded issues.",
      {},
      async () => ok(JSON.stringify(await store.listIssues(projectId))),
    ),

    tool(
      "apply_edit",
      "Apply an edit to the document (replace a char range). Gated by permission mode.",
      { char_start: z.number().int(), char_end: z.number().int(), replacement: z.string() },
      async ({ char_start, char_end, replacement }) => {
        await store.applyEdit(projectId, char_start, char_end, replacement);
        return ok(`applied edit [${char_start},${char_end}]`);
      },
    ),

    tool(
      "export_document",
      "Return the current document text for saving/export.",
      {},
      async () => ok(await store.getDocument(projectId)),
    ),
  ];
}
