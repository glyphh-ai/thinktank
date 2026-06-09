// Auth primitives on Node crypto + jose. Passwords use scrypt; tokens/PINs are
// random; stored secrets are sha256-hashed; access tokens are HS256 JWTs.

import { randomBytes, randomInt, scryptSync, timingSafeEqual, createHash } from "node:crypto";
import { SignJWT, jwtVerify } from "jose";
import { config } from "../config";

export function hashPassword(password: string): string {
  const salt = randomBytes(16);
  const dk = scryptSync(password, salt, 64);
  return `scrypt$${salt.toString("base64")}$${dk.toString("base64")}`;
}

export function verifyPassword(password: string, encoded: string): boolean {
  const parts = encoded.split("$");
  if (parts.length !== 3 || parts[0] !== "scrypt") return false;
  const salt = Buffer.from(parts[1]!, "base64");
  const expected = Buffer.from(parts[2]!, "base64");
  const dk = scryptSync(password, salt, expected.length);
  return expected.length === dk.length && timingSafeEqual(expected, dk);
}

export function generatePin(digits = 6): string {
  let pin = "";
  for (let i = 0; i < digits; i++) pin += randomInt(0, 10).toString();
  return pin;
}

export function generateToken(bytes = 32): string {
  return randomBytes(bytes).toString("base64url");
}

export function hashSecret(secret: string): string {
  return createHash("sha256").update(secret).digest("hex");
}

export function verifySecret(secret: string, secretHash: string): boolean {
  const a = Buffer.from(hashSecret(secret));
  const b = Buffer.from(secretHash);
  return a.length === b.length && timingSafeEqual(a, b);
}

const jwtKey = () => new TextEncoder().encode(config.jwtSecret);

export async function signAccessToken(claims: { sub: string; email: string }): Promise<string> {
  return new SignJWT(claims)
    .setProtectedHeader({ alg: "HS256", typ: "JWT" })
    .setIssuedAt()
    .setExpirationTime(`${config.accessTokenTtlSeconds}s`)
    .sign(jwtKey());
}

export async function verifyAccessToken(token: string): Promise<{ sub: string; email: string }> {
  const { payload } = await jwtVerify(token, jwtKey());
  return { sub: String(payload.sub), email: String(payload.email) };
}
