import { body, handle, json, requireUser } from "@/lib/http";
import { createProject, listProjects } from "@/lib/projects";

export const runtime = "nodejs";

export const GET = (req: Request) =>
  handle(async () => {
    const user = await requireUser(req);
    return json(await listProjects(user.id));
  });

export const POST = (req: Request) =>
  handle(async () => {
    const user = await requireUser(req);
    const { name, text } = await body<{ name: string; text: string }>(req);
    return json(await createProject(user.id, name, { text }), 201);
  });
