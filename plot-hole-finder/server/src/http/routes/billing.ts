// Billing routes: Stripe checkout (authenticated) + webhook (raw body verified).

import type { FastifyInstance, FastifyRequest } from "fastify";
import { createCheckout, handleWebhook } from "@/lib/billing";
import { ValidationError } from "@/lib/errors";
import type { PlanTier } from "@/lib/plans";
import { jsonBody, requireUser } from "../context";

export async function billingRoutes(app: FastifyInstance) {
  app.post("/api/billing/checkout", async (req) => {
    const user = await requireUser(req);
    const { tier } = jsonBody<{ tier: PlanTier }>(req);
    if (tier !== "basic" && tier !== "pro") throw new ValidationError("tier must be basic or pro");
    return { url: await createCheckout(user, tier) };
  });

  app.post("/api/billing/webhook", async (req: FastifyRequest & { rawBody?: Buffer }) => {
    const signature = (req.headers["stripe-signature"] as string) ?? "";
    // Stripe signature verification needs the exact raw bytes (see app.ts parser).
    const payload = req.rawBody ? req.rawBody.toString("utf8") : JSON.stringify(req.body ?? {});
    return { received: await handleWebhook(payload, signature) };
  });
}
