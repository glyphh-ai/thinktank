import { currentPlan } from "@/lib/billing";
import { handle, json, requireUser } from "@/lib/http";

export const runtime = "nodejs";

export const GET = (req: Request) =>
  handle(async () => {
    const user = await requireUser(req);
    return json({ id: user.id, email: user.email, plan: await currentPlan(user.id) });
  });
