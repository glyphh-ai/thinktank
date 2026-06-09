// Runtime configuration from the environment.

function env(key: string, fallback = ""): string {
  return process.env[key] ?? fallback;
}

export const config = {
  anthropicApiKey: env("ANTHROPIC_API_KEY"),
  anthropicModel: env("ANTHROPIC_MODEL", "claude-opus-4-8"),
  databaseUrl: env("DATABASE_URL"),
  redisUrl: env("REDIS_URL"),
  jwtSecret: env("JWT_SECRET", "dev-insecure-change-me"),
  accessTokenTtlSeconds: Number(env("ACCESS_TOKEN_TTL", "900")),
  refreshTokenTtlSeconds: Number(env("REFRESH_TOKEN_TTL", String(60 * 60 * 24 * 30))),
  emailPinTtlSeconds: Number(env("EMAIL_PIN_TTL", "600")),
  passwordResetTtlSeconds: Number(env("PASSWORD_RESET_TTL", "3600")),
  appUrl: env("APP_URL", "http://localhost:3000"),
  emailBackend: env("EMAIL_BACKEND", "console"), // console | resend | smtp
  resendApiKey: env("RESEND_API_KEY"),
  emailFrom: env("EMAIL_FROM", "noreply@plot-hole-finder.dev"),
  stripeSecretKey: env("STRIPE_SECRET_KEY"),
  stripeWebhookSecret: env("STRIPE_WEBHOOK_SECRET"),
  stripePriceBasic: env("STRIPE_PRICE_BASIC"),
  stripePricePro: env("STRIPE_PRICE_PRO"),
  billingEnabled: env("BILLING_BACKEND", "disabled") === "stripe",
  jobLeaseSeconds: Number(env("JOB_LEASE_SECONDS", "900")),
  // Agent run guardrails (cost/turn caps so a massive-doc pass can't run away).
  agentMaxTurns: Number(env("AGENT_MAX_TURNS", "200")),
  agentMaxBudgetUsd: Number(env("AGENT_MAX_BUDGET_USD", "25")),
  // Rate limits (fixed window).
  rateLimitWindowSeconds: Number(env("RATE_LIMIT_WINDOW", "60")),
  authRateLimit: Number(env("AUTH_RATE_LIMIT", "20")),
  agentRateLimit: Number(env("AGENT_RATE_LIMIT", "30")),
  mcpRateLimit: Number(env("MCP_RATE_LIMIT", "120")),
  // Allowed Host header values for MCP DNS-rebinding protection (comma-sep).
  // Empty disables the check (dev). Set in prod, e.g. "api.example.com".
  mcpAllowedHosts: env("MCP_ALLOWED_HOSTS").split(",").map((h) => h.trim()).filter(Boolean),
} as const;
