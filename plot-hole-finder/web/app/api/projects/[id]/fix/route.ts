import { getDb } from "@/lib/db";
import { currentPlan } from "@/lib/billing";
import { Forbidden } from "@/lib/errors";
import { enqueue } from "@/lib/jobs";
import { getProject } from "@/lib/projects";
import { limitsFor } from "@/lib/plans";
import { handle, json, requireUser } from "@/lib/http";

export const runtime = "nodejs";

export const POST = (req: Request, { params }: { params: Promise<{ id: string }> }) =>
  handle(async () => {
    const user = await requireUser(req);
    const { id } = await params;
    await getProject(user.id, id);
    if (!limitsFor(await currentPlan(user.id)).canFix) {
      throw new Forbidden("fix-this-for-me requires the Pro plan");
    }
    const job = await enqueue(getDb(), user.id, "fix", { projectId: id });
    return json({ jobId: job.id, status: job.status }, 202);
  });
