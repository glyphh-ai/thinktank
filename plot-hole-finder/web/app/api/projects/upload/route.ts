import { handle, json, requireUser } from "@/lib/http";
import { createProject } from "@/lib/projects";
import { ValidationError } from "@/lib/errors";

export const runtime = "nodejs";

export const POST = (req: Request) =>
  handle(async () => {
    const user = await requireUser(req);
    const form = await req.formData();
    const name = String(form.get("name") ?? "");
    const file = form.get("file");
    if (!(file instanceof File)) throw new ValidationError("file is required");
    const data = new Uint8Array(await file.arrayBuffer());
    return json(await createProject(user.id, name, { filename: file.name, data }), 201);
  });
