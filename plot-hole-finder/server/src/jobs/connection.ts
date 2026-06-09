// Shared ioredis connection for BullMQ. BullMQ requires `maxRetriesPerRequest:
// null` on the connection it uses for blocking commands.

import IORedis, { type RedisOptions } from "ioredis";
import { config } from "@/lib/config";

export const ANALYSIS_QUEUE = "analysis";

let _conn: IORedis | undefined;

/** Lazily-created Redis connection suitable for BullMQ (queues + workers). */
export function jobConnection(): IORedis {
  if (!_conn) {
    if (!config.redisUrl) {
      throw new Error("REDIS_URL is required for the job queue");
    }
    const opts: RedisOptions = { maxRetriesPerRequest: null, enableReadyCheck: false };
    _conn = new IORedis(config.redisUrl, opts);
  }
  return _conn;
}

export async function closeJobConnection(): Promise<void> {
  if (_conn) {
    await _conn.quit().catch(() => {});
    _conn = undefined;
  }
}
