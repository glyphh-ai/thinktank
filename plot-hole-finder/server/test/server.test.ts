// HTTP-layer tests: routing, validation, error mapping, rate limiting, readiness,
// and MCP auth rejection. These use Fastify's `inject` (no real network) and do
// not require Postgres/Redis — the cache/rate-limiter fall back to in-memory.

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { FastifyInstance } from "fastify";
import { buildApp } from "../src/http/app";

let app: FastifyInstance;

beforeAll(async () => {
  app = await buildApp({ logger: false });
  await app.ready();
});

afterAll(async () => {
  await app.close();
});

describe("http server", () => {
  it("serves liveness", async () => {
    const res = await app.inject({ method: "GET", url: "/health" });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ ok: true });
  });

  it("readiness reports dependency checks (503 without a database)", async () => {
    const res = await app.inject({ method: "GET", url: "/ready" });
    expect(res.statusCode).toBe(503);
    const body = res.json();
    expect(body.ready).toBe(false);
    expect(body.checks).toHaveProperty("db");
  });

  it("rejects unauthenticated access", async () => {
    const res = await app.inject({
      method: "GET",
      url: "/api/auth/me",
      headers: { "x-forwarded-for": "10.0.0.1" },
    });
    expect(res.statusCode).toBe(401);
    expect(res.json().error).toBe("unauthorized");
  });

  it("validates request bodies with zod", async () => {
    const res = await app.inject({
      method: "POST",
      url: "/api/auth/register",
      headers: { "x-forwarded-for": "10.0.0.2" },
      payload: { email: "not-an-email", password: "x" },
    });
    expect(res.statusCode).toBe(422);
    expect(res.json().error).toBe("validation_error");
  });

  it("rate-limits sensitive endpoints per client IP", async () => {
    const ip = "10.0.0.99";
    let limited = false;
    // authRateLimit defaults to 20/60s; the 21st request from one IP should 429.
    for (let i = 0; i < 25; i++) {
      const res = await app.inject({
        method: "POST",
        url: "/api/auth/login",
        headers: { "x-forwarded-for": ip },
        payload: { email: "a@b.com", password: "secret123" },
      });
      if (res.statusCode === 429) {
        expect(res.json().error).toBe("rate_limited");
        limited = true;
        break;
      }
    }
    expect(limited).toBe(true);
  });

  it("MCP endpoint rejects requests without a bearer token (JSON-RPC error)", async () => {
    const res = await app.inject({
      method: "POST",
      url: "/mcp",
      headers: {
        "x-forwarded-for": "10.0.0.3",
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
      },
      payload: { jsonrpc: "2.0", id: 1, method: "tools/list" },
    });
    const body = res.json();
    expect(body.jsonrpc).toBe("2.0");
    expect(body.error.code).toBe(-32001);
    expect(body.error.message).toMatch(/bearer/i);
  });
});
