// BullMQ analysis queue: enqueue analyze/fix runs and read back durable status.
// Massive documents are processed here (not inline in an HTTP request) so a long
// rolling-memory pass survives request timeouts and can be retried/observed.

import { Queue, type JobsOptions } from "bullmq";
import { ANALYSIS_QUEUE, jobConnection } from "./connection";

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

export async function enqueueAnalysis(
  data: AnalysisJobData,
  opts?: JobsOptions,
): Promise<{ jobId: string; state: string }> {
  const job = await analysisQueue().add(data.type, data, opts);
  return { jobId: job.id!, state: await job.getState() };
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
