// Request helpers shared across Fastify routes.

import type { FastifyReply, FastifyRequest } from "fastify";
import { AppError, Unauthorized, ValidationError } from "@/lib/errors";
import { userFromAuthHeader } from "@/lib/auth/service";

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
