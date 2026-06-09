// Auth routes: register -> verify email -> login -> refresh -> logout, plus
// password reset and the authenticated /me endpoint. Sensitive endpoints are
// rate-limited per client IP and validated with zod.

import type { FastifyInstance } from "fastify";
import { z } from "zod";
import {
  login, logout, refresh, register, requestPasswordReset,
  resetPassword, verifyEmail,
} from "@/lib/auth/service";
import { currentPlan } from "@/lib/billing";
import { config } from "@/lib/config";
import { clientIp, enforceRateLimit, parseBody, requireUser } from "../context";

const emailSchema = z.string().email().max(320);
const credentials = z.object({ email: emailSchema, password: z.string().min(1).max(512) });

export async function authRoutes(app: FastifyInstance) {
  // Throttle every auth endpoint by client IP (credential stuffing / abuse).
  app.addHook("onRequest", async (req) => {
    await enforceRateLimit("auth", clientIp(req), config.authRateLimit);
  });

  app.post("/api/auth/register", async (req, reply) => {
    const { email, password } = parseBody(req, credentials);
    await register(email, password);
    return reply.code(201).send({ message: "registered; check your email for a verification code" });
  });

  app.post("/api/auth/verify-email", async (req) => {
    const { email, pin } = parseBody(req, z.object({ email: emailSchema, pin: z.string().min(1).max(32) }));
    await verifyEmail(email, pin);
    return { message: "email verified" };
  });

  app.post("/api/auth/login", async (req) => {
    const { email, password } = parseBody(req, credentials);
    return login(email, password);
  });

  app.post("/api/auth/refresh", async (req) => {
    const { refreshToken } = parseBody(req, z.object({ refreshToken: z.string().min(1) }));
    return refresh(refreshToken);
  });

  app.post("/api/auth/logout", async (req) => {
    const { refreshToken } = parseBody(req, z.object({ refreshToken: z.string().min(1) }));
    await logout(refreshToken);
    return { message: "logged out" };
  });

  app.post("/api/auth/forgot-password", async (req) => {
    const { email } = parseBody(req, z.object({ email: emailSchema }));
    await requestPasswordReset(email);
    return { message: "if the account exists, a reset link was sent" };
  });

  app.post("/api/auth/reset-password", async (req) => {
    const { token, password } = parseBody(
      req,
      z.object({ token: z.string().min(1), password: z.string().min(8).max(512) }),
    );
    await resetPassword(token, password);
    return { message: "password reset; please log in" };
  });

  app.get("/api/auth/me", async (req) => {
    const user = await requireUser(req);
    return { id: user.id, email: user.email, plan: await currentPlan(user.id) };
  });
}
