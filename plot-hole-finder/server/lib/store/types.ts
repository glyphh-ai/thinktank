// Domain store the agent tools operate over. The in-memory implementation lets
// the agent loop run and be tested now; a Drizzle/Postgres implementation
// (same interface) is the next step for the "rock-solid" backend.

export type Severity = "low" | "medium" | "high";
export type IssueCategory =
  | "continuity"
  | "timeline"
  | "character"
  | "knowledge"
  | "broken_reference"
  | "sequencing";

export interface MemoryFact {
  entity: string; // surface name, the retrieval key
  key: string; // attribute / relation
  value: string;
  kind: "state" | "event";
  chunkIndex: number;
  // For events that participate in the timeline:
  storyTime?: number | null; // in-world order (flashbacks differ from narrative order)
  charStart?: number;
}

export interface Issue {
  id: string;
  category: IssueCategory;
  severity: Severity;
  title: string;
  description: string;
  evidence: { role: "state_a" | "state_b" | "context"; charStart: number; charEnd: number; quote: string }[];
  suggestedFix?: string;
  status: "open" | "dismissed" | "fixed";
}

export interface TimelineEvent {
  id: string;
  summary: string;
  entity?: string;
  narrativeTime: number; // position in the text (char offset / chunk index)
  storyTime: number; // inferred in-world order
  charStart: number;
  charEnd: number;
}

export interface TimelineNote {
  id: string;
  charStart: number;
  charEnd: number;
  text: string;
  kind: "note" | "plot_hole" | "sequencing";
  issueId?: string;
}

export interface ProjectState {
  id: string;
  documentText: string;
  memory: MemoryFact[];
  issues: Issue[];
  timeline: TimelineEvent[];
  notes: TimelineNote[];
}

export interface Store {
  getDocument(projectId: string): Promise<string>;
  applyEdit(projectId: string, charStart: number, charEnd: number, replacement: string): Promise<void>;

  getMemory(projectId: string, entities?: string[]): Promise<MemoryFact[]>;
  addMemory(projectId: string, facts: MemoryFact[]): Promise<void>;

  listIssues(projectId: string): Promise<Issue[]>;
  addIssue(projectId: string, issue: Omit<Issue, "id" | "status">): Promise<Issue>;
  updateIssue(projectId: string, id: string, patch: Partial<Issue>): Promise<void>;

  getTimeline(projectId: string): Promise<TimelineEvent[]>;
  addTimelineEvent(projectId: string, event: Omit<TimelineEvent, "id">): Promise<TimelineEvent>;

  listNotes(projectId: string): Promise<TimelineNote[]>;
  addNote(projectId: string, note: Omit<TimelineNote, "id">): Promise<TimelineNote>;
}
