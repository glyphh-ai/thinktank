import { handleWebhook } from "@/lib/billing";
import { handle, json } from "@/lib/http";

export const runtime = "nodejs";

export const POST = (req: Request) =>
  handle(async () => {
    const signature = req.headers.get("stripe-signature") ?? "";
    const payload = await req.text(); // raw body required for signature verification
    return json({ received: await handleWebhook(payload, signature) });
  });
