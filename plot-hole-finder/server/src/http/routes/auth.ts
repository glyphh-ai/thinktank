// Auth routes: register -> verify email -> login -> refresh -> logout, plus
// password reset and the authenticated /me endpoint.

import type { FastifyInstance } from "fastify";
import {
  login, logout, refresh, register, requestPasswordReset,
  resetPassword, verifyEmail,
} from "@/lib/auth/service";
import { currentPlan } from "@/lib/billing";
import { jsonBody, requireUser } from "../context";

export async function authRoutes(app: FastifyInstance) {
  app.post("/api/auth/register", async (req, reply) => {
    const { email, password } = jsonBody<{ email: string; password: string }>(req);
    await register(email, password);
    return reply.code(201).send({ message: "registered; check your email for a verification code" });
  });

  app.post("/api/auth/verify-email", async (req) => {
    const { email, pin } = jsonBody<{ email: string; pin: string }>(req);
    await verifyEmail(email, pin);
    return { message: "email verified" };
  });

  app.post("/api/auth/login", async (req) => {
    const { email, password } = jsonBody<{ email: string; password: string }>(req);
    return login(email, password);
  });

  app.post("/api/auth/refresh", async (req) => {
    const { refreshToken } = jsonBody<{ refreshToken: string }>(req);
    return refresh(refreshToken);
  });

  app.post("/api/auth/logout", async (req) => {
    const { refreshToken } = jsonBody<{ refreshToken: string }>(req);
    await logout(refreshToken);
    return { message: "logged out" };
  });

  app.post("/api/auth/forgot-password", async (req) => {
    const { email } = jsonBody<{ email: string }>(req);
    await requestPasswordReset(email);
    return { message: "if the account exists, a reset link was sent" };
  });

  app.post("/api/auth/reset-password", async (req) => {
    const { token, password } = jsonBody<{ token: string; password: string }>(req);
    await resetPassword(token, password);
    return { message: "password reset; please log in" };
  });

  app.get("/api/auth/me", async (req) => {
    const user = await requireUser(req);
    return { id: user.id, email: user.email, plan: await currentPlan(user.id) };
  });
}
