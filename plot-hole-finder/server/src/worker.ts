// Worker process entrypoint: `pnpm worker`. Runs the BullMQ analysis worker in
// its own process so heavy agent passes are isolated from the API server and can
// be scaled independently.

import { closeDb } from "@/lib/db";
import { closeRedis } from "@/lib/redis";
import { startWorker } from "./jobs/worker";
import { closeJobConnection } from "./jobs/connection";

const worker = startWorker();
console.log(`[worker] analysis worker started (pid ${process.pid})`);

async function shutdown(signal: string) {
  console.log(`[worker] received ${signal}, draining...`);
  await worker.close(); // stop taking new jobs, finish in-flight
  await Promise.allSettled([closeJobConnection(), closeRedis(), closeDb()]);
  process.exit(0);
}

process.on("SIGTERM", () => void shutdown("SIGTERM"));
process.on("SIGINT", () => void shutdown("SIGINT"));
