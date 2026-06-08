import { login } from "@/lib/auth/service";
import { body, handle, json } from "@/lib/http";

export const runtime = "nodejs";

export const POST = (req: Request) =>
  handle(async () => {
    const { email, password } = await body<{ email: string; password: string }>(req);
    return json(await login(email, password));
  });
