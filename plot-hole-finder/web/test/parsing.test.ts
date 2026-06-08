import { readFile } from "node:fs/promises";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { extractText, stripHtml, UnsupportedDocumentError, wordCount } from "../lib/parsing";

const fix = (name: string) => readFile(path.join(__dirname, "fixtures", name));
const bytes = (s: string) => new TextEncoder().encode(s);

describe("parsing", () => {
  it("decodes plain text and counts words", async () => {
    expect(await extractText("a.txt", bytes("Hello world"))).toBe("Hello world");
    expect(wordCount("one two three")).toBe(3);
  });

  it("strips html", () => {
    const t = stripHtml("<html><style>x{}</style><h1>T</h1><p>Elena&#39;s eyes were green.</p></html>");
    expect(t).toContain("Elena's eyes were green.");
    expect(t).not.toContain("<");
  });

  it("extracts csv/json as text", async () => {
    expect(await extractText("d.csv", bytes("name,age\nalice,30"))).toContain("alice");
  });

  it("extracts a real PDF", async () => {
    const t = await extractText("sample.pdf", new Uint8Array(await fix("sample.pdf")));
    expect(t.toLowerCase()).toContain("green");
  });

  it("extracts a real DOCX (incl. table cells)", async () => {
    const t = await extractText("sample.docx", new Uint8Array(await fix("sample.docx")));
    expect(t.toLowerCase()).toContain("green");
    expect(t).toContain("Exhibit A");
  });

  it("rejects unsupported types", async () => {
    await expect(extractText("a.zip", bytes("PK"))).rejects.toBeInstanceOf(UnsupportedDocumentError);
  });
});
