// Background worker: claims durable jobs and runs the agent over a project.
// Deployment-agnostic — run multiple replicas; claiming is safe (SKIP LOCKED).

import { getDb, type DB } from "./db";
import { config } from "./config";
import * as q from "./jobs";
import { DrizzleStore } from "./store/drizzle";
import { runAgent } from "./agent/runner";

const ANALYZE_PROMPT =
  "Analyze this entire document for plot holes, continuity errors, and " +
  "timeline/sequencing problems. Read it fully (page through with read_document), " +
  "remember facts as you go, recall before judging, and record_issue for every " +
  "genuine problem with evidence spans. Also record_timeline_event for events and " +
  "flag sequencing issues. Do not edit the document.";

const FIX_PROMPT =
  "Review the recorded issues (list_issues) and the document. For each fixable " +
  "issue, propose and apply_edit a concise correction that keeps the author's voice.";

export class Worker {
  private stopped = false;
  constructor(private db: DB = getDb()) {}

  stop() { this.stopped = true; }

  async runAvailable(limit = 100): Promise<number> {
    let n = 0;
    while (n < limit && !this.stopped) {
      const job = await q.claimNext(this.db, config.jobLeaseSeconds);
      if (!job) break;
      await this.run(job);
      n++;
    }
    return n;
  }

  async runForever(pollMs = 2000): Promise<void> {
    while (!this.stopped) {
      await q.reclaimStale(this.db);
      if ((await this.runAvailable()) === 0) await new Promise((r) => setTimeout(r, pollMs));
    }
  }

  private async run(job: q.JobRow): Promise<void> {
    try {
      if (job.type === "analyze") await this.analyze(job);
      else if (job.type === "fix") await this.fix(job);
      else await q.complete(this.db, job.id, { skipped: true });
    } catch (err) {
      await q.fail(this.db, job, err instanceof Error ? err.message : String(err));
    }
  }

  private async analyze(job: q.JobRow): Promise<void> {
    const store = new DrizzleStore(this.db);
    const projectId = job.projectId!;
    for await (const ev of runAgent({ store, projectId, prompt: ANALYZE_PROMPT, mode: "ask",
      model: config.anthropicModel })) {
      if (ev.type === "tool_use" || ev.type === "result") {
        await q.heartbeat(this.db, job.id, config.jobLeaseSeconds, undefined, ev.type);
      }
    }
    const issues = await store.listIssues(projectId);
    const timeline = await store.getTimeline(projectId);
    await q.complete(this.db, job.id, { issues: issues.length, timelineEvents: timeline.length });
  }

  private async fix(job: q.JobRow): Promise<void> {
    const store = new DrizzleStore(this.db);
    const projectId = job.projectId!;
    let edits = 0;
    for await (const ev of runAgent({ store, projectId, prompt: FIX_PROMPT, mode: "auto",
      model: config.anthropicModel })) {
      if (ev.type === "tool_use") {
        if (ev.name === "apply_edit") edits++;
        await q.heartbeat(this.db, job.id, config.jobLeaseSeconds, undefined, ev.type);
      }
    }
    await q.complete(this.db, job.id, { edits });
  }
}

export async function runWorker(): Promise<void> {
  // eslint-disable-next-line no-console
  console.log("worker started");
  await new Worker().runForever();
}
