import { AppError } from "./errors";
import { userFromAuthHeader } from "./auth/service";

export function json(data: unknown, status = 200): Response {
  return Response.json(data, { status });
}

export async function handle(fn: () => Promise<Response>): Promise<Response> {
  try {
    return await fn();
  } catch (err) {
    if (err instanceof AppError) return json({ error: err.code, message: err.message }, err.statusCode);
    console.error(err);
    return json({ error: "internal", message: "internal error" }, 500);
  }
}

export function requireUser(req: Request) {
  return userFromAuthHeader(req.headers.get("authorization"));
}

export async function body<T = Record<string, unknown>>(req: Request): Promise<T> {
  return (await req.json().catch(() => ({}))) as T;
}
