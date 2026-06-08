import { handle, json, requireUser } from "@/lib/http";
import { getProject } from "@/lib/projects";
import { DrizzleStore } from "@/lib/store/drizzle";

export const runtime = "nodejs";

export const GET = (req: Request, { params }: { params: Promise<{ id: string }> }) =>
  handle(async () => {
    const user = await requireUser(req);
    const { id } = await params;
    await getProject(user.id, id);
    const store = new DrizzleStore();
    return json({ timeline: await store.getTimeline(id), notes: await store.listNotes(id) });
  });
