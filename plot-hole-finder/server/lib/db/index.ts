import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import * as schema from "./schema";

export type DB = ReturnType<typeof drizzle<typeof schema>>;

let _db: DB | undefined;

// Lazy so importing this module (e.g. in tests) doesn't require a live database.
export function getDb(): DB {
  if (!_db) {
    const url = process.env.DATABASE_URL;
    if (!url) throw new Error("DATABASE_URL is required");
    _db = drizzle(postgres(url, { max: 10 }), { schema });
  }
  return _db;
}

export { schema };
