import { verifyEmail } from "@/lib/auth/service";
import { body, handle, json } from "@/lib/http";

export const runtime = "nodejs";

export const POST = (req: Request) =>
  handle(async () => {
    const { email, pin } = await body<{ email: string; pin: string }>(req);
    await verifyEmail(email, pin);
    return json({ message: "email verified" });
  });
