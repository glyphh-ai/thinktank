// Request helpers shared across Fastify routes.

import type { FastifyReply, FastifyRequest } from "fastify";
import type { z } from "zod";
import { AppError, Forbidden, Unauthorized, ValidationError } from "@/lib/errors";
import { userFromAuthHeader } from "@/lib/auth/service";
import { config } from "@/lib/config";
import { rateLimit } from "@/lib/redis";

export interface AuthedUser { id: string; email: string }

/** Authenticate via the Authorization: Bearer header, or throw Unauthorized. */
export function requireUser(req: FastifyRequest): Promise<AuthedUser> {
  return userFromAuthHeader(req.headers.authorization ?? null);
}

/** Validate that a JSON body is an object and return it typed. */
export function jsonBody<T>(req: FastifyRequest): T {
  const b = req.body;
  if (b === null || typeof b !== "object") throw new ValidationError("expected a JSON object body");
  return b as T;
}

/** Parse + validate the JSON body against a zod schema, or throw ValidationError. */
export function parseBody<S extends z.ZodTypeAny>(req: FastifyRequest, schema: S): z.infer<S> {
  const result = schema.safeParse(req.body ?? {});
  if (!result.success) {
    const first = result.error.issues[0];
    const path = first?.path.join(".");
    throw new ValidationError(`${path ? `${path}: ` : ""}${first?.message ?? "invalid request body"}`);
  }
  return result.data;
}

/** Best-effort client IP for unauthenticated rate-limit buckets. */
export function clientIp(req: FastifyRequest): string {
  const fwd = req.headers["x-forwarded-for"];
  if (typeof fwd === "string" && fwd.length) return fwd.split(",")[0]!.trim();
  return req.ip;
}

/**
 * Fixed-window rate limit; throws Forbidden (429) when exceeded. `bucket` scopes
 * the limit (e.g. "auth", "agent") and `id` identifies the caller (ip or userId).
 */
export async function enforceRateLimit(bucket: string, id: string, limit: number): Promise<void> {
  const ok = await rateLimit(`${bucket}:${id}`, limit, config.rateLimitWindowSeconds);
  if (!ok) {
    const err = new Forbidden("rate limit exceeded; slow down");
    err.statusCode = 429;
    err.code = "rate_limited";
    throw err;
  }
}

/** Maps thrown AppErrors to clean JSON responses; everything else is a 500. */
export function sendError(reply: FastifyReply, err: unknown): FastifyReply {
  if (err instanceof AppError) {
    return reply.code(err.statusCode).send({ error: err.code, message: err.message });
  }
  reply.log.error(err);
  const message = err instanceof Error ? err.message : "internal error";
  return reply.code(500).send({ error: "internal_error", message });
}

export { Unauthorized };
