// Extract plain text from any uploaded document. Native parsers for the common
// formats; LibreOffice (headless) as the universal fallback for the office family.

import { spawn } from "node:child_process";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

const TEXT_EXT = new Set([".txt", ".md", ".markdown", ".text", ".log", ".csv", ".tsv", ".json", ".xml", ""]);
const HTML_EXT = new Set([".html", ".htm", ".xhtml"]);
const OFFICE_EXT = new Set([".doc", ".odt", ".rtf", ".ppt", ".pptx", ".xls", ".xlsx", ".odp", ".ods"]);
export const SUPPORTED_EXTENSIONS = new Set([...TEXT_EXT, ...HTML_EXT, ...OFFICE_EXT, ".pdf", ".docx", ".epub"]);

export class UnsupportedDocumentError extends Error {
  code = "unsupported_document";
}

const ext = (filename: string) => path.extname(filename || "").toLowerCase();

const MIME: Record<string, string> = {
  ".txt": "text/plain", ".md": "text/markdown", ".csv": "text/csv", ".json": "application/json",
  ".html": "text/html", ".htm": "text/html", ".xml": "application/xml", ".pdf": "application/pdf",
  ".epub": "application/epub+zip", ".rtf": "application/rtf", ".doc": "application/msword",
  ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
};

export const mimeFor = (filename: string): string => MIME[ext(filename)] ?? "application/octet-stream";

export async function extractText(filename: string, data: Uint8Array): Promise<string> {
  const e = ext(filename);
  if (TEXT_EXT.has(e)) return new TextDecoder("utf-8", { fatal: false }).decode(data);
  if (HTML_EXT.has(e)) return stripHtml(new TextDecoder().decode(data));
  if (e === ".pdf") return fromPdf(data);
  if (e === ".docx") return fromDocx(data);
  if (e === ".epub") return fromEpub(data);
  if (OFFICE_EXT.has(e)) return fromLibreOffice(data, e);
  throw new UnsupportedDocumentError(`unsupported file type '${e}'`);
}

export function stripHtml(html: string): string {
  const noScripts = html.replace(/<(script|style)\b[\s\S]*?<\/\1>/gi, " ");
  const noTags = noScripts.replace(/<[^>]+>/g, " ");
  const unescaped = noTags
    .replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'");
  return unescaped.replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
}

async function fromPdf(data: Uint8Array): Promise<string> {
  const { extractText: extract, getDocumentProxy } = await import("unpdf");
  const pdf = await getDocumentProxy(data);
  const { text } = await extract(pdf, { mergePages: true });
  return text.trim();
}

async function fromDocx(data: Uint8Array): Promise<string> {
  const mammoth = (await import("mammoth")).default;
  const { value } = await mammoth.extractRawText({ buffer: Buffer.from(data) });
  return value.trim();
}

async function fromEpub(data: Uint8Array): Promise<string> {
  const JSZip = (await import("jszip")).default;
  const zip = await JSZip.loadAsync(data);
  const names = Object.keys(zip.files)
    .filter((n) => /\.(x?html?|htm)$/i.test(n)).sort();
  const parts: string[] = [];
  for (const name of names) parts.push(stripHtml(await zip.files[name]!.async("string")));
  const text = parts.filter(Boolean).join("\n\n");
  if (!text.trim()) throw new UnsupportedDocumentError("could not extract text from .epub");
  return text;
}

async function fromLibreOffice(data: Uint8Array, e: string): Promise<string> {
  const dir = await mkdtemp(path.join(tmpdir(), "phf-"));
  try {
    const src = path.join(dir, `input${e}`);
    await writeFile(src, data);
    await new Promise<void>((resolve, reject) => {
      const p = spawn("soffice", ["--headless", "--convert-to", "txt:Text", "--outdir", dir, src],
        { env: { ...process.env, HOME: dir } });
      const timer = setTimeout(() => { p.kill(); reject(new Error("libreoffice timeout")); }, 180_000);
      p.on("error", reject);
      p.on("close", () => { clearTimeout(timer); resolve(); });
    }).catch(() => {
      throw new UnsupportedDocumentError(`reading ${e} requires a working LibreOffice`);
    });
    const out = (await readdir(dir)).find((f) => f.endsWith(".txt"));
    if (!out) throw new UnsupportedDocumentError(`could not convert ${e} file`);
    return await readFile(path.join(dir, out), "utf-8");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

export function wordCount(text: string): number {
  return (text.match(/\b\w+\b/g) ?? []).length;
}
