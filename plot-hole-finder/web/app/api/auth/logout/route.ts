import { logout } from "@/lib/auth/service";
import { body, handle, json } from "@/lib/http";

export const runtime = "nodejs";

export const POST = (req: Request) =>
  handle(async () => {
    const { refreshToken } = await body<{ refreshToken: string }>(req);
    await logout(refreshToken);
    return json({ message: "logged out" });
  });
