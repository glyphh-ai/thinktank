// Billing: plan resolution + Stripe (checkout / portal / idempotent webhook).

import { eq } from "drizzle-orm";
import { getDb } from "./db";
import { subscriptions, users } from "./db/schema";
import { config } from "./config";
import { AppError } from "./errors";
import type { PlanTier } from "./plans";
import { PLANS } from "./plans";

const ACTIVE = new Set(["active", "trialing"]);

export async function currentPlan(userId: string): Promise<PlanTier> {
  const [sub] = await getDb().select().from(subscriptions).where(eq(subscriptions.userId, userId)).limit(1);
  return sub && ACTIVE.has(sub.status) ? (sub.plan as PlanTier) : "free";
}

export async function getOrCreateSubscription(userId: string) {
  const db = getDb();
  const [sub] = await db.select().from(subscriptions).where(eq(subscriptions.userId, userId)).limit(1);
  if (sub) return sub;
  const [created] = await db.insert(subscriptions).values({ userId, plan: "free" }).returning();
  return created!;
}

function stripe() {
  if (!config.billingEnabled) throw new AppError("billing is not enabled");
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const Stripe = require("stripe");
  return new Stripe(config.stripeSecretKey);
}

const priceFor = (tier: PlanTier): string =>
  tier === "pro" ? config.stripePricePro : config.stripePriceBasic;

export async function createCheckout(user: { id: string; email: string }, tier: PlanTier): Promise<string> {
  const s = stripe();
  const db = getDb();
  const [u] = await db.select().from(users).where(eq(users.id, user.id)).limit(1);
  let customerId = u?.stripeCustomerId ?? null;
  if (!customerId) {
    const customer = await s.customers.create({ email: user.email, metadata: { user_id: user.id } });
    customerId = customer.id;
    await db.update(users).set({ stripeCustomerId: customerId }).where(eq(users.id, user.id));
  }
  const session = await s.checkout.sessions.create({
    mode: "subscription", customer: customerId,
    line_items: [{ price: priceFor(tier), quantity: 1 }],
    success_url: `${config.appUrl}/billing/success?session_id={CHECKOUT_SESSION_ID}`,
    cancel_url: `${config.appUrl}/billing`,
    metadata: { user_id: user.id, tier },
  });
  return session.url as string;
}

export async function handleWebhook(payload: string, signature: string): Promise<string> {
  const s = stripe();
  const event = s.webhooks.constructEvent(payload, signature, config.stripeWebhookSecret);
  const db = getDb();
  const obj = event.data.object as Record<string, any>;

  const subForCustomer = async (customerId: string) => {
    const [u] = await db.select().from(users).where(eq(users.stripeCustomerId, customerId)).limit(1);
    if (!u) return null;
    return getOrCreateSubscription(u.id);
  };

  if (event.type === "checkout.session.completed") {
    const sub = await subForCustomer(obj.customer);
    if (sub) {
      const tier = (obj.metadata?.tier as PlanTier) ?? "basic";
      await db.update(subscriptions).set({
        plan: tier, status: "active", stripeSubscriptionId: obj.subscription ?? null,
      }).where(eq(subscriptions.id, sub.id));
    }
  } else if (event.type.startsWith("customer.subscription.")) {
    const sub = await subForCustomer(obj.customer);
    if (sub) {
      const deleted = event.type === "customer.subscription.deleted";
      const priceId = obj.items?.data?.[0]?.price?.id as string | undefined;
      const tier = (Object.values(PLANS).find((p) => p.stripePriceEnv && config[p.stripePriceEnv] === priceId)?.tier) ?? sub.plan as PlanTier;
      await db.update(subscriptions).set({
        status: deleted ? "canceled" : obj.status, plan: deleted ? "free" : tier,
        stripeSubscriptionId: obj.id, stripePriceId: priceId ?? null,
        cancelAtPeriodEnd: !!obj.cancel_at_period_end,
      }).where(eq(subscriptions.id, sub.id));
    }
  }
  return event.type;
}
