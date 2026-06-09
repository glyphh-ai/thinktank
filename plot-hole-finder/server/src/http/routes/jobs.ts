// Job status route: read BullMQ job state/progress/result, scoped to the owner.

import type { FastifyInstance } from "fastify";
import { NotFound } from "@/lib/errors";
import { jobStatus } from "../../jobs/queue";
import { requireUser } from "../context";

export async function jobRoutes(app: FastifyInstance) {
  app.get("/api/jobs/:id", async (req) => {
    const user = await requireUser(req);
    const { id } = req.params as { id: string };
    const job = await jobStatus(id);
    if (!job || job.userId !== user.id) throw new NotFound("job not found");
    return {
      id: job.id,
      type: job.type,
      status: job.state,
      progress: job.progress,
      step: job.step,
      error: job.error,
      result: job.result,
    };
  });
}
