import { resetPassword } from "@/lib/auth/service";
import { body, handle, json } from "@/lib/http";

export const runtime = "nodejs";

export const POST = (req: Request) =>
  handle(async () => {
    const { token, password } = await body<{ token: string; password: string }>(req);
    await resetPassword(token, password);
    return json({ message: "password reset; please log in" });
  });
