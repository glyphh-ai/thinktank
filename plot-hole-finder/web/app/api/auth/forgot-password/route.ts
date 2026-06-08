import { requestPasswordReset } from "@/lib/auth/service";
import { body, handle, json } from "@/lib/http";

export const runtime = "nodejs";

export const POST = (req: Request) =>
  handle(async () => {
    const { email } = await body<{ email: string }>(req);
    await requestPasswordReset(email);
    return json({ message: "if the account exists, a reset link was sent" });
  });
