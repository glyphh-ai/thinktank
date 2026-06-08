import { createCheckout } from "@/lib/billing";
import { ValidationError } from "@/lib/errors";
import { body, handle, json, requireUser } from "@/lib/http";
import type { PlanTier } from "@/lib/plans";

export const runtime = "nodejs";

export const POST = (req: Request) =>
  handle(async () => {
    const user = await requireUser(req);
    const { tier } = await body<{ tier: PlanTier }>(req);
    if (tier !== "basic" && tier !== "pro") throw new ValidationError("tier must be basic or pro");
    return json({ url: await createCheckout(user, tier) });
  });
