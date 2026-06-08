// Durable, DB-backed job queue. Lease-based claiming (FOR UPDATE SKIP LOCKED on
// Postgres), retries with backoff, stale reclaim. Workers run as separate
// processes for scale.

import { and, eq, isNotNull, lte, sql } from "drizzle-orm";
import { getDb, type DB } from "./db";
import { jobs } from "./db/schema";

export type JobType = "analyze" | "fix";
export type JobRow = typeof jobs.$inferSelect;

export async function enqueue(
  db: DB, userId: string, type: JobType,
  opts: { projectId?: string; payload?: Record<string, unknown>; maxAttempts?: number } = {},
): Promise<JobRow> {
  const [row] = await db.insert(jobs).values({
    userId, projectId: opts.projectId ?? null, type, status: "queued",
    payload: opts.payload ?? {}, maxAttempts: opts.maxAttempts ?? 3,
  }).returning();
  return row!;
}

export async function claimNext(db: DB, leaseSeconds: number): Promise<JobRow | null> {
  // Atomically claim the oldest eligible queued job. SKIP LOCKED makes it safe
  // across concurrent workers on Postgres.
  const rows = await db.execute<JobRow>(sql`
    UPDATE ${jobs} SET status = 'running', attempts = attempts + 1,
      lease_expires_at = now() + (${leaseSeconds} * interval '1 second'), updated_at = now()
    WHERE id = (
      SELECT id FROM ${jobs}
      WHERE status = 'queued' AND scheduled_at <= now()
      ORDER BY scheduled_at, created_at
      FOR UPDATE SKIP LOCKED LIMIT 1
    )
    RETURNING *`);
  return (rows as unknown as { rows?: JobRow[] }).rows?.[0] ?? (Array.isArray(rows) ? rows[0] ?? null : null);
}

export async function heartbeat(db: DB, id: string, leaseSeconds: number, progress?: number, step?: string) {
  await db.update(jobs).set({
    leaseExpiresAt: new Date(Date.now() + leaseSeconds * 1000),
    ...(progress !== undefined ? { progress: Math.max(0, Math.min(1, progress)) } : {}),
    ...(step !== undefined ? { step } : {}),
  }).where(eq(jobs.id, id));
}

export async function complete(db: DB, id: string, result: Record<string, unknown>) {
  await db.update(jobs).set({ status: "succeeded", progress: 1, result, error: null, leaseExpiresAt: null })
    .where(eq(jobs.id, id));
}

export async function fail(db: DB, job: JobRow, error: string, backoffBase = 5) {
  if (job.attempts < job.maxAttempts) {
    const delay = backoffBase * 2 ** (job.attempts - 1);
    await db.update(jobs).set({
      status: "queued", error, leaseExpiresAt: null,
      scheduledAt: new Date(Date.now() + delay * 1000),
    }).where(eq(jobs.id, job.id));
  } else {
    await db.update(jobs).set({ status: "failed", error, leaseExpiresAt: null }).where(eq(jobs.id, job.id));
  }
}

export async function reclaimStale(db: DB): Promise<number> {
  const res = await db.update(jobs).set({ status: "queued", leaseExpiresAt: null })
    .where(and(eq(jobs.status, "running"), isNotNull(jobs.leaseExpiresAt), lte(jobs.leaseExpiresAt, new Date())));
  return (res as unknown as { count?: number }).count ?? 0;
}

export async function getJob(db: DB, id: string): Promise<JobRow | null> {
  const [row] = await db.select().from(jobs).where(eq(jobs.id, id)).limit(1);
  return row ?? null;
}
