// Runtime configuration from the environment.

function env(key: string, fallback = ""): string {
  return process.env[key] ?? fallback;
}

export const config = {
  anthropicApiKey: env("ANTHROPIC_API_KEY"),
  anthropicModel: env("ANTHROPIC_MODEL", "claude-opus-4-8"),
  databaseUrl: env("DATABASE_URL"),
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
} as const;
