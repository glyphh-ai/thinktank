// Authentication: register -> email PIN verify -> login (JWT) -> refresh rotation
// -> logout -> forgot/reset password. Stores only hashes.

import { and, desc, eq, isNull } from "drizzle-orm";
import { getDb } from "../db";
import { refreshTokens, subscriptions, users, verificationTokens } from "../db/schema";
import { config } from "../config";
import { Conflict, Forbidden, Unauthorized, ValidationError } from "../errors";
import { passwordResetEmail, sendEmail, verificationPinEmail } from "../email";
import {
  generatePin, generateToken, hashPassword, hashSecret, signAccessToken,
  verifyAccessToken, verifyPassword, verifySecret,
} from "./crypto";

const MAX_PIN_ATTEMPTS = 5;
const now = () => new Date();
const expired = (d: Date) => d.getTime() < Date.now();

function normalizeEmail(email: string): string {
  const e = email.trim().toLowerCase();
  if (!e.includes("@") || !e.split("@")[1]?.includes(".")) throw new ValidationError("invalid email address");
  return e;
}

export interface TokenPair { accessToken: string; refreshToken: string }

export async function register(email: string, password: string): Promise<string> {
  email = normalizeEmail(email);
  if (password.length < 8) throw new ValidationError("password must be at least 8 characters");
  const db = getDb();
  const [existing] = await db.select({ id: users.id }).from(users).where(eq(users.email, email)).limit(1);
  if (existing) throw new Conflict("an account with this email already exists");

  const [user] = await db.insert(users).values({ email, passwordHash: hashPassword(password) }).returning();
  await db.insert(subscriptions).values({ userId: user!.id, plan: "free" });
  await issueEmailPin(user!.id, email);
  return user!.id;
}

export async function issueEmailPin(userId: string, email: string): Promise<void> {
  const pin = generatePin();
  await getDb().insert(verificationTokens).values({
    userId, kind: "email_pin", secretHash: hashSecret(pin),
    expiresAt: new Date(Date.now() + config.emailPinTtlSeconds * 1000),
  });
  await sendEmail(verificationPinEmail(email, pin));
}

export async function verifyEmail(email: string, pin: string): Promise<void> {
  email = normalizeEmail(email);
  const db = getDb();
  const [user] = await db.select().from(users).where(eq(users.email, email)).limit(1);
  if (!user) throw new Unauthorized("invalid code");
  if (user.emailVerified) return;

  const [token] = await db.select().from(verificationTokens)
    .where(and(eq(verificationTokens.userId, user.id), eq(verificationTokens.kind, "email_pin"),
      isNull(verificationTokens.consumedAt)))
    .orderBy(desc(verificationTokens.createdAt)).limit(1);
  if (!token || expired(token.expiresAt)) throw new Unauthorized("code expired or not found");
  if (token.attempts >= MAX_PIN_ATTEMPTS) throw new Forbidden("too many attempts; request a new code");

  await db.update(verificationTokens).set({ attempts: token.attempts + 1 }).where(eq(verificationTokens.id, token.id));
  if (!verifySecret(pin, token.secretHash)) throw new Unauthorized("invalid code");

  await db.update(verificationTokens).set({ consumedAt: now() }).where(eq(verificationTokens.id, token.id));
  await db.update(users).set({ emailVerified: true }).where(eq(users.id, user.id));
}

export async function login(email: string, password: string): Promise<TokenPair> {
  email = normalizeEmail(email);
  const db = getDb();
  const [user] = await db.select().from(users).where(eq(users.email, email)).limit(1);
  if (!user || !verifyPassword(password, user.passwordHash)) throw new Unauthorized("invalid email or password");
  if (user.status !== "active") throw new Forbidden("account is not active");
  if (!user.emailVerified) throw new Forbidden("email not verified");
  return issueTokens(user.id, user.email);
}

export async function issueTokens(userId: string, email: string): Promise<TokenPair> {
  const accessToken = await signAccessToken({ sub: userId, email });
  const raw = generateToken();
  await getDb().insert(refreshTokens).values({
    userId, tokenHash: hashSecret(raw),
    expiresAt: new Date(Date.now() + config.refreshTokenTtlSeconds * 1000),
  });
  return { accessToken, refreshToken: raw };
}

export async function refresh(rawRefresh: string): Promise<TokenPair> {
  const db = getDb();
  const [rec] = await db.select().from(refreshTokens)
    .where(eq(refreshTokens.tokenHash, hashSecret(rawRefresh))).limit(1);
  if (!rec || rec.revokedAt || expired(rec.expiresAt)) throw new Unauthorized("invalid refresh token");
  await db.update(refreshTokens).set({ revokedAt: now() }).where(eq(refreshTokens.id, rec.id)); // rotate
  const [user] = await db.select().from(users).where(eq(users.id, rec.userId)).limit(1);
  if (!user) throw new Unauthorized("invalid refresh token");
  return issueTokens(user.id, user.email);
}

export async function logout(rawRefresh: string): Promise<void> {
  await getDb().update(refreshTokens).set({ revokedAt: now() })
    .where(eq(refreshTokens.tokenHash, hashSecret(rawRefresh)));
}

export async function requestPasswordReset(email: string): Promise<void> {
  email = normalizeEmail(email);
  const db = getDb();
  const [user] = await db.select().from(users).where(eq(users.email, email)).limit(1);
  if (!user) return; // don't reveal existence
  const raw = generateToken();
  await db.insert(verificationTokens).values({
    userId: user.id, kind: "password_reset", secretHash: hashSecret(raw),
    expiresAt: new Date(Date.now() + config.passwordResetTtlSeconds * 1000),
  });
  await sendEmail(passwordResetEmail(email, `${config.appUrl}/reset-password?token=${raw}`));
}

export async function resetPassword(rawToken: string, newPassword: string): Promise<void> {
  if (newPassword.length < 8) throw new ValidationError("password must be at least 8 characters");
  const db = getDb();
  const [token] = await db.select().from(verificationTokens)
    .where(and(eq(verificationTokens.kind, "password_reset"),
      eq(verificationTokens.secretHash, hashSecret(rawToken)), isNull(verificationTokens.consumedAt)))
    .limit(1);
  if (!token || expired(token.expiresAt)) throw new Unauthorized("invalid or expired reset token");
  await db.update(verificationTokens).set({ consumedAt: now() }).where(eq(verificationTokens.id, token.id));
  await db.update(users).set({ passwordHash: hashPassword(newPassword) }).where(eq(users.id, token.userId));
  await db.update(refreshTokens).set({ revokedAt: now() })
    .where(and(eq(refreshTokens.userId, token.userId), isNull(refreshTokens.revokedAt)));
}

export async function userFromAuthHeader(header: string | null): Promise<{ id: string; email: string }> {
  if (!header?.toLowerCase().startsWith("bearer ")) throw new Unauthorized("missing bearer token");
  let claims;
  try {
    claims = await verifyAccessToken(header.slice(7).trim());
  } catch {
    throw new Unauthorized("invalid token");
  }
  const [user] = await getDb().select().from(users).where(eq(users.id, claims.sub)).limit(1);
  if (!user || user.status !== "active") throw new Unauthorized("invalid token");
  return { id: user.id, email: user.email };
}
