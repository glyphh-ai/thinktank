// BullMQ worker: executes analyze/fix jobs by running the Claude Agent SDK
// rolling-memory loop (runAgent) over the durable Drizzle store. This is what
// makes massive documents (200k–1M+ tokens) tractable: the long pass runs in a
// background worker with retries, progress, and automatic lock renewal rather
// than inside an HTTP request.

import { UnrecoverableError, Worker, type Job } from "bullmq";
import { config } from "@/lib/config";
import { DrizzleStore } from "@/lib/store/drizzle";
import { runAgent } from "@/lib/agent/runner";
import { ANALYSIS_QUEUE, jobConnection } from "./connection";
import { cancelKey, type AnalysisJobData } from "./queue";

// Extends UnrecoverableError so a cancelled job is not retried by BullMQ.
export class JobCancelledError extends UnrecoverableError {
  constructor() { super("job cancelled"); this.name = "JobCancelledError"; }
}

const ANALYZE_PROMPT =
  "Analyze this document for plot holes, continuity errors, and timeline/sequencing problems. " +
  "Read it incrementally, remember durable facts, recall them to cross-check, and record every issue " +
  "with evidence spans plus the reconstructed timeline. Do not modify the document.";

const FIX_PROMPT =
  "Find and FIX plot holes, continuity errors, and timeline/sequencing problems in this document. " +
  "Read it incrementally with rolling memory, record each issue with evidence, then apply minimal, " +
  "consistent edits that resolve them. Keep the author's voice. Summarize what you changed.";

async function processJob(job: Job<AnalysisJobData>) {
  const { type, projectId, prompt, model } = job.data;
  const store = new DrizzleStore();
  const redis = jobConnection();

  // Cooperative cancellation: trips the AbortSignal runAgent forwards to interrupt().
  const ac = new AbortController();
  const cancelled = async () => (await redis.get(cancelKey(job.id!))) !== null;

  let tools = 0;
  let issues = 0;
  await job.updateProgress({ pct: 1, step: "starting" });

  for await (const ev of runAgent({
    store,
    projectId,
    prompt: prompt ?? (type === "fix" ? FIX_PROMPT : ANALYZE_PROMPT),
    // analyze never edits (read-only); fix runs unattended so edits are allowed.
    mode: type === "fix" ? "auto" : "ask",
    model: model ?? config.anthropicModel,
    // Hard guardrails so a massive-doc pass can't run away on turns or cost.
    maxTurns: config.agentMaxTurns,
    maxBudgetUsd: config.agentMaxBudgetUsd,
    signal: ac.signal,
  })) {
    if (ev.type === "tool_use") {
      tools++;
      if (ev.name === "record_issue") issues++;
      // updateProgress also renews the job lock, keeping long passes alive.
      await job.updateProgress({
        pct: Math.min(95, 5 + tools),
        step: `${ev.name} (${tools} tool calls, ${issues} issues)`,
      });
      // Check for a cancellation request between tool calls.
      if (await cancelled()) {
        ac.abort();
        await redis.del(cancelKey(job.id!));
        throw new JobCancelledError();
      }
    } else if (ev.type === "error") {
      throw new Error(ev.message);
    }
  }

  await job.updateProgress({ pct: 100, step: "done" });
  return {
    issues: await store.listIssues(projectId),
    timeline: await store.getTimeline(projectId),
    notes: await store.listNotes(projectId),
  };
}

export function startWorker(): Worker<AnalysisJobData> {
  const worker = new Worker<AnalysisJobData>(ANALYSIS_QUEUE, processJob, {
    connection: jobConnection(),
    concurrency: Number(process.env.WORKER_CONCURRENCY ?? "2"),
    // Massive-document passes are long; give the lock plenty of headroom.
    lockDuration: config.jobLeaseSeconds * 1000,
    stalledInterval: 30_000,
  });

  worker.on("failed", (job, err) => {
    console.error(`[worker] job ${job?.id} (${job?.name}) failed:`, err.message);
  });
  worker.on("completed", (job) => {
    console.log(`[worker] job ${job.id} (${job.name}) completed`);
  });

  return worker;
}
