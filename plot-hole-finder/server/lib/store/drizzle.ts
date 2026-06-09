// Durable Store backed by Postgres (Drizzle). Same interface as InMemoryStore,
// so the agent tools and tests don't change.

import { and, desc, eq, inArray } from "drizzle-orm";
import { getDb, type DB } from "../db";
import {
  documents,
  issueEvidence,
  issues,
  memoryFacts,
  notes,
  timelineEvents,
} from "../db/schema";
import type {
  Issue,
  IssueCategory,
  MemoryFact,
  Severity,
  Store,
  TimelineEvent,
  TimelineNote,
} from "./types";

export class DrizzleStore implements Store {
  constructor(private db: DB = getDb()) {}

  async getDocument(projectId: string): Promise<string> {
    const [doc] = await this.db.select({ content: documents.content })
      .from(documents).where(eq(documents.projectId, projectId)).limit(1);
    return doc?.content ?? "";
  }

  async applyEdit(projectId: string, charStart: number, charEnd: number, replacement: string) {
    const current = await this.getDocument(projectId);
    const next = current.slice(0, charStart) + replacement + current.slice(charEnd);
    await this.db.update(documents).set({ content: next }).where(eq(documents.projectId, projectId));
  }

  async getMemory(projectId: string, entities?: string[]): Promise<MemoryFact[]> {
    const where = entities?.length
      ? and(eq(memoryFacts.projectId, projectId), inArray(memoryFacts.entity, entities))
      : eq(memoryFacts.projectId, projectId);
    const rows = await this.db.select().from(memoryFacts).where(where)
      .orderBy(desc(memoryFacts.chunkIndex)).limit(500);
    return rows.map((r) => ({
      entity: r.entity, key: r.key, value: r.value, kind: r.kind as MemoryFact["kind"],
      chunkIndex: r.chunkIndex, storyTime: r.storyTime, charStart: r.charStart ?? undefined,
    }));
  }

  async addMemory(projectId: string, facts: MemoryFact[]) {
    if (!facts.length) return;
    await this.db.insert(memoryFacts).values(facts.map((f) => ({
      projectId, entity: f.entity, key: f.key, value: f.value, kind: f.kind,
      chunkIndex: f.chunkIndex, storyTime: f.storyTime ?? null, charStart: f.charStart ?? null,
    })));
  }

  async listIssues(projectId: string): Promise<Issue[]> {
    const rows = await this.db.select().from(issues).where(eq(issues.projectId, projectId))
      .orderBy(desc(issues.createdAt));
    if (!rows.length) return [];
    const ev = await this.db.select().from(issueEvidence)
      .where(inArray(issueEvidence.issueId, rows.map((r) => r.id)));
    return rows.map((r) => ({
      id: r.id, category: r.category as IssueCategory, severity: r.severity as Severity,
      title: r.title, description: r.description, suggestedFix: r.suggestedFix ?? undefined,
      status: r.status as Issue["status"],
      evidence: ev.filter((e) => e.issueId === r.id).map((e) => ({
        role: e.role as Issue["evidence"][number]["role"],
        charStart: e.charStart, charEnd: e.charEnd, quote: e.quote,
      })),
    }));
  }

  async addIssue(projectId: string, issue: Omit<Issue, "id" | "status">): Promise<Issue> {
    const [row] = await this.db.insert(issues).values({
      projectId, category: issue.category, severity: issue.severity, title: issue.title,
      description: issue.description, suggestedFix: issue.suggestedFix ?? null,
    }).returning();
    if (issue.evidence.length) {
      await this.db.insert(issueEvidence).values(issue.evidence.map((e) => ({
        issueId: row!.id, role: e.role, charStart: e.charStart, charEnd: e.charEnd, quote: e.quote,
      })));
    }
    return { ...issue, id: row!.id, status: "open" };
  }

  async updateIssue(projectId: string, id: string, patch: Partial<Issue>) {
    const set: Record<string, unknown> = {};
    if (patch.suggestedFix !== undefined) set.suggestedFix = patch.suggestedFix;
    if (patch.status !== undefined) set.status = patch.status;
    if (Object.keys(set).length) await this.db.update(issues).set(set).where(eq(issues.id, id));
  }

  async getTimeline(projectId: string): Promise<TimelineEvent[]> {
    const rows = await this.db.select().from(timelineEvents)
      .where(eq(timelineEvents.projectId, projectId)).orderBy(timelineEvents.storyTime);
    return rows.map((r) => ({
      id: r.id, summary: r.summary, entity: r.entity ?? undefined,
      narrativeTime: r.narrativeTime, storyTime: r.storyTime,
      charStart: r.charStart, charEnd: r.charEnd,
    }));
  }

  async addTimelineEvent(projectId: string, e: Omit<TimelineEvent, "id">): Promise<TimelineEvent> {
    const [row] = await this.db.insert(timelineEvents).values({
      projectId, summary: e.summary, entity: e.entity ?? null, narrativeTime: e.narrativeTime,
      storyTime: e.storyTime, charStart: e.charStart, charEnd: e.charEnd,
    }).returning();
    return { ...e, id: row!.id };
  }

  async listNotes(projectId: string): Promise<TimelineNote[]> {
    const rows = await this.db.select().from(notes).where(eq(notes.projectId, projectId));
    return rows.map((r) => ({
      id: r.id, charStart: r.charStart, charEnd: r.charEnd, text: r.text,
      kind: r.kind as TimelineNote["kind"], issueId: r.issueId ?? undefined,
    }));
  }

  async addNote(projectId: string, note: Omit<TimelineNote, "id">): Promise<TimelineNote> {
    const [row] = await this.db.insert(notes).values({
      projectId, charStart: note.charStart, charEnd: note.charEnd, text: note.text,
      kind: note.kind, issueId: note.issueId ?? null,
    }).returning();
    return { ...note, id: row!.id };
  }
}
