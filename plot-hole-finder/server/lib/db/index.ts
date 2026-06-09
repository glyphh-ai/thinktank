import { sql } from "drizzle-orm";
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import * as schema from "./schema";

export type DB = ReturnType<typeof drizzle<typeof schema>>;

let _client: ReturnType<typeof postgres> | undefined;
let _db: DB | undefined;

// Lazy so importing this module (e.g. in tests) doesn't require a live database.
export function getDb(): DB {
  if (!_db) {
    const url = process.env.DATABASE_URL;
    if (!url) throw new Error("DATABASE_URL is required");
    _client = postgres(url, { max: 10 });
    _db = drizzle(_client, { schema });
  }
  return _db;
}

/** Liveness check used by the readiness probe. */
export async function pingDb(): Promise<void> {
  await getDb().execute(sql`select 1`);
}

export async function closeDb(): Promise<void> {
  if (_client) {
    await _client.end({ timeout: 5 }).catch(() => {});
    _client = undefined;
    _db = undefined;
  }
}

export { schema };
