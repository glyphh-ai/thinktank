// BullMQ analysis queue: enqueue analyze/fix runs and read back durable status.
// Massive documents are processed here (not inline in an HTTP request) so a long
// rolling-memory pass survives request timeouts and can be retried/observed.

import { Queue, type JobsOptions } from "bullmq";
import { ANALYSIS_QUEUE, closeJobConnection, jobConnection } from "./connection";

export type JobType = "analyze" | "fix";

export interface AnalysisJobData {
  type: JobType;
  userId: string;
  projectId: string;
  prompt?: string;
  model?: string;
}

export interface JobStatus {
  id: string;
  type: JobType;
  state: string; // waiting | active | completed | failed | delayed | ...
  progress: number;
  step?: string;
  result?: unknown;
  error?: string;
  userId: string;
  projectId: string;
}

let _queue: Queue<AnalysisJobData> | undefined;

export function analysisQueue(): Queue<AnalysisJobData> {
  if (!_queue) {
    _queue = new Queue<AnalysisJobData>(ANALYSIS_QUEUE, {
      connection: jobConnection(),
      defaultJobOptions: {
        attempts: 3,
        backoff: { type: "exponential", delay: 5000 },
        removeOnComplete: { age: 60 * 60 * 24 * 7, count: 1000 },
        removeOnFail: { age: 60 * 60 * 24 * 30 },
      },
    });
  }
  return _queue;
}

// Redis key a worker polls to cooperatively cancel an in-flight run.
export const cancelKey = (jobId: string) => `cancel:job:${jobId}`;

/** Close the queue and its Redis connection (graceful shutdown). */
export async function closeQueue(): Promise<void> {
  if (_queue) {
    await _queue.close().catch(() => {});
    _queue = undefined;
  }
  await closeJobConnection();
}

/**
 * Enqueue an analyze/fix run, deduplicating against any in-flight job for the
 * same project+type. Runs are expensive (a full rolling-memory pass over a
 * massive doc), so we never stack a second one — we return the existing job.
 */
export async function enqueueAnalysis(
  data: AnalysisJobData,
  opts?: JobsOptions,
): Promise<{ jobId: string; state: string; deduped: boolean }> {
  const queue = analysisQueue();
  const inflight = await queue.getJobs(["active", "waiting", "delayed", "paused"]);
  const existing = inflight.find(
    (j) => j.data.projectId === data.projectId && j.data.type === data.type,
  );
  if (existing) return { jobId: existing.id!, state: await existing.getState(), deduped: true };

  const job = await queue.add(data.type, data, opts);
  return { jobId: job.id!, state: await job.getState(), deduped: false };
}

/** Reads job status. Ownership is enforced by the caller against `userId`. */
export async function jobStatus(jobId: string): Promise<JobStatus | null> {
  const job = await analysisQueue().getJob(jobId);
  if (!job) return null;
  const progress = typeof job.progress === "number" ? job.progress : 0;
  const step = job.progress && typeof job.progress === "object"
    ? (job.progress as { step?: string }).step
    : undefined;
  return {
    id: job.id!,
    type: (job.data.type ?? job.name) as JobType,
    state: await job.getState(),
    progress: typeof job.progress === "object" ? ((job.progress as { pct?: number }).pct ?? 0) : progress,
    step,
    result: job.returnvalue,
    error: job.failedReason || undefined,
    userId: job.data.userId,
    projectId: job.data.projectId,
  };
}

/**
 * Cancel a job. Waiting/delayed jobs are removed outright; an active job is
 * signalled via a Redis flag the worker polls (BullMQ can't force-kill a running
 * processor). Returns null if the job doesn't exist, else its owner for the
 * caller's ownership check.
 */
export async function cancelJob(jobId: string): Promise<{ userId: string; state: string } | null> {
  const queue = analysisQueue();
  const job = await queue.getJob(jobId);
  if (!job) return null;
  const userId = job.data.userId;
  const state = await job.getState();
  if (state === "active") {
    // Cooperative: worker checks this flag each iteration and aborts.
    await jobConnection().set(cancelKey(jobId), "1", "EX", 3600);
  } else if (state === "waiting" || state === "delayed" || state === "prioritized") {
    await job.remove();
  }
  return { userId, state };
}
