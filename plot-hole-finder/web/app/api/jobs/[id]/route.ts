import { getDb } from "@/lib/db";
import { NotFound } from "@/lib/errors";
import { getJob } from "@/lib/jobs";
import { handle, json, requireUser } from "@/lib/http";

export const runtime = "nodejs";

export const GET = (req: Request, { params }: { params: Promise<{ id: string }> }) =>
  handle(async () => {
    const user = await requireUser(req);
    const { id } = await params;
    const job = await getJob(getDb(), id);
    if (!job || job.userId !== user.id) throw new NotFound("job not found");
    return json({
      id: job.id, type: job.type, status: job.status, progress: job.progress,
      step: job.step, error: job.error, result: job.result,
    });
  });
