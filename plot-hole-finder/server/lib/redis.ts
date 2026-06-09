// Redis client (lazy) + a cache/rate-limit layer with an in-memory fallback so
// the app runs without Redis in dev but uses Redis when REDIS_URL is set.

import IORedis, { type Redis } from "ioredis";
import { config } from "./config";

let _redis: Redis | null | undefined;

export function getRedis(): Redis | null {
  if (_redis === undefined) {
    _redis = config.redisUrl
      ? new IORedis(config.redisUrl, { maxRetriesPerRequest: 3, lazyConnect: false })
      : null;
  }
  return _redis ?? null;
}

export async function closeRedis(): Promise<void> {
  if (_redis) {
    await _redis.quit().catch(() => {});
  }
  _redis = undefined;
  _cache = undefined;
}

export interface Cache {
  get(key: string): Promise<string | null>;
  set(key: string, value: string, ttlSeconds?: number): Promise<void>;
  del(key: string): Promise<void>;
  incr(key: string, ttlSeconds?: number): Promise<number>;
}

class RedisCache implements Cache {
  constructor(private r: Redis) {}
  async get(k: string) { return this.r.get(k); }
  async set(k: string, v: string, ttl?: number) {
    if (ttl) await this.r.set(k, v, "EX", ttl);
    else await this.r.set(k, v);
  }
  async del(k: string) { await this.r.del(k); }
  async incr(k: string, ttl?: number) {
    const n = await this.r.incr(k);
    if (n === 1 && ttl) await this.r.expire(k, ttl);
    return n;
  }
}

class MemoryCache implements Cache {
  private store = new Map<string, { v: string; exp?: number }>();
  private live(k: string) {
    const it = this.store.get(k);
    if (!it) return null;
    if (it.exp && it.exp < Date.now()) { this.store.delete(k); return null; }
    return it.v;
  }
  async get(k: string) { return this.live(k); }
  async set(k: string, v: string, ttl?: number) {
    this.store.set(k, { v, exp: ttl ? Date.now() + ttl * 1000 : undefined });
  }
  async del(k: string) { this.store.delete(k); }
  async incr(k: string, ttl?: number) {
    const n = Number(this.live(k) ?? 0) + 1;
    const existing = this.store.get(k);
    this.store.set(k, { v: String(n), exp: existing?.exp ?? (ttl ? Date.now() + ttl * 1000 : undefined) });
    return n;
  }
}

let _cache: Cache | undefined;

export function getCache(): Cache {
  if (!_cache) {
    const r = getRedis();
    _cache = r ? new RedisCache(r) : new MemoryCache();
  }
  return _cache;
}

/** Fixed-window rate limit. Returns true if the action is allowed. */
export async function rateLimit(key: string, limit: number, windowSeconds: number): Promise<boolean> {
  const n = await getCache().incr(`rl:${key}`, windowSeconds);
  return n <= limit;
}
