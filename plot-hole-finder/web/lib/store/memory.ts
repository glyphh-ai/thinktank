// In-memory Store implementation. Swappable for a Drizzle/Postgres one (same
// interface). Used to run/test the agent loop before the DB port.

import { randomUUID } from "node:crypto";
import type {
  Issue,
  MemoryFact,
  ProjectState,
  Store,
  TimelineEvent,
  TimelineNote,
} from "./types";

export class InMemoryStore implements Store {
  private projects = new Map<string, ProjectState>();

  constructor(seed?: { id: string; documentText: string }) {
    if (seed) {
      this.projects.set(seed.id, {
        id: seed.id,
        documentText: seed.documentText,
        memory: [],
        issues: [],
        timeline: [],
        notes: [],
      });
    }
  }

  private get(projectId: string): ProjectState {
    const p = this.projects.get(projectId);
    if (!p) throw new Error(`unknown project: ${projectId}`);
    return p;
  }

  async getDocument(projectId: string) {
    return this.get(projectId).documentText;
  }

  async applyEdit(projectId: string, charStart: number, charEnd: number, replacement: string) {
    const p = this.get(projectId);
    p.documentText = p.documentText.slice(0, charStart) + replacement + p.documentText.slice(charEnd);
  }

  async getMemory(projectId: string, entities?: string[]) {
    const all = this.get(projectId).memory;
    if (!entities?.length) return all;
    const set = new Set(entities.map((e) => e.toLowerCase()));
    return all.filter((f) => set.has(f.entity.toLowerCase()));
  }

  async addMemory(projectId: string, facts: MemoryFact[]) {
    this.get(projectId).memory.push(...facts);
  }

  async listIssues(projectId: string) {
    return this.get(projectId).issues;
  }

  async addIssue(projectId: string, issue: Omit<Issue, "id" | "status">) {
    const full: Issue = { ...issue, id: randomUUID(), status: "open" };
    this.get(projectId).issues.push(full);
    return full;
  }

  async updateIssue(projectId: string, id: string, patch: Partial<Issue>) {
    const issue = this.get(projectId).issues.find((i) => i.id === id);
    if (issue) Object.assign(issue, patch);
  }

  async getTimeline(projectId: string) {
    return this.get(projectId).timeline;
  }

  async addTimelineEvent(projectId: string, event: Omit<TimelineEvent, "id">) {
    const full: TimelineEvent = { ...event, id: randomUUID() };
    this.get(projectId).timeline.push(full);
    return full;
  }

  async listNotes(projectId: string) {
    return this.get(projectId).notes;
  }

  async addNote(projectId: string, note: Omit<TimelineNote, "id">) {
    const full: TimelineNote = { ...note, id: randomUUID() };
    this.get(projectId).notes.push(full);
    return full;
  }
}
