import { describe, expect, it } from "vitest";
import {
  generatePin, generateToken, hashPassword, hashSecret, signAccessToken,
  verifyAccessToken, verifyPassword, verifySecret,
} from "../lib/auth/crypto";
import { limitsFor } from "../lib/plans";

describe("password hashing", () => {
  it("verifies the right password and rejects the wrong one", () => {
    const h = hashPassword("supersecret");
    expect(h.startsWith("scrypt$")).toBe(true);
    expect(verifyPassword("supersecret", h)).toBe(true);
    expect(verifyPassword("wrong", h)).toBe(false);
  });
});

describe("secrets and pins", () => {
  it("pins are 6 digits; secrets verify", () => {
    expect(generatePin()).toMatch(/^\d{6}$/);
    const tok = generateToken();
    expect(verifySecret(tok, hashSecret(tok))).toBe(true);
    expect(verifySecret("nope", hashSecret(tok))).toBe(false);
  });
});

describe("access tokens (HS256 JWT)", () => {
  it("round-trips claims", async () => {
    const jwt = await signAccessToken({ sub: "u1", email: "a@b.com" });
    const claims = await verifyAccessToken(jwt);
    expect(claims.sub).toBe("u1");
    expect(claims.email).toBe("a@b.com");
  });

  it("rejects a tampered token", async () => {
    const jwt = await signAccessToken({ sub: "u1", email: "a@b.com" });
    await expect(verifyAccessToken(jwt + "x")).rejects.toBeTruthy();
  });
});

describe("plans", () => {
  it("free can't fix; pro can and allows huge docs", () => {
    expect(limitsFor("free").canFix).toBe(false);
    expect(limitsFor("pro").canFix).toBe(true);
    expect(limitsFor("pro").maxWordsPerProject).toBeGreaterThan(1_000_000);
  });
});
