// Projects: create from pasted text or any uploaded document, with plan limits.

import { createHash } from "node:crypto";
import { count, desc, eq } from "drizzle-orm";
import { getDb } from "./db";
import { documents, projects } from "./db/schema";
import { currentPlan } from "./billing";
import { NotFound, PlanLimitExceeded, ValidationError } from "./errors";
import { extractText, mimeFor, wordCount } from "./parsing";
import { limitsFor } from "./plans";

export async function createProject(
  userId: string, name: string,
  source: { text: string } | { filename: string; data: Uint8Array },
): Promise<{ id: string; wordCount: number }> {
  const db = getDb();
  const limits = limitsFor(await currentPlan(userId));

  const [countRow] = await db.select({ value: count() }).from(projects).where(eq(projects.userId, userId));
  if ((countRow?.value ?? 0) >= limits.maxProjects) {
    throw new PlanLimitExceeded(`plan limit reached: ${limits.maxProjects} project(s). Upgrade to add more.`);
  }

  let text: string;
  let filename: string;
  let data: Uint8Array;
  if ("text" in source) {
    if (!source.text.trim()) throw new ValidationError("pasted text is empty");
    text = source.text;
    filename = "pasted.txt";
    data = new TextEncoder().encode(text);
  } else {
    if (!source.data?.length) throw new ValidationError("uploaded file is empty");
    filename = source.filename || "upload.txt";
    data = source.data;
    text = await extractText(filename, data);
    if (!text.trim()) throw new ValidationError("no readable text could be extracted from the document");
  }

  const wc = wordCount(text);
  if (wc > limits.maxWordsPerProject) {
    throw new PlanLimitExceeded(
      `document has ${wc.toLocaleString()} words; your plan allows ${limits.maxWordsPerProject.toLocaleString()}. Upgrade for larger documents.`);
  }

  const [project] = await db.insert(projects).values({ userId, name: name.trim() || "Untitled", wordCount: wc }).returning();
  await db.insert(documents).values({
    projectId: project!.id, filename, mimeType: mimeFor(filename), content: text,
    byteSize: data.length, wordCount: wc, sha256: createHash("sha256").update(data).digest("hex"),
  });
  return { id: project!.id, wordCount: wc };
}

export async function listProjects(userId: string) {
  return getDb().select().from(projects).where(eq(projects.userId, userId)).orderBy(desc(projects.createdAt));
}

export async function getProject(userId: string, id: string) {
  const [p] = await getDb().select().from(projects).where(eq(projects.id, id)).limit(1);
  if (!p || p.userId !== userId) throw new NotFound("project not found");
  return p;
}
