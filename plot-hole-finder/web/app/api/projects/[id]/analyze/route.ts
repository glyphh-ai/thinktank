import { getDb } from "@/lib/db";
import { enqueue } from "@/lib/jobs";
import { getProject } from "@/lib/projects";
import { handle, json, requireUser } from "@/lib/http";

export const runtime = "nodejs";

export const POST = (req: Request, { params }: { params: Promise<{ id: string }> }) =>
  handle(async () => {
    const user = await requireUser(req);
    const { id } = await params;
    await getProject(user.id, id); // ownership check
    const job = await enqueue(getDb(), user.id, "analyze", { projectId: id });
    return json({ jobId: job.id, status: job.status }, 202);
  });
