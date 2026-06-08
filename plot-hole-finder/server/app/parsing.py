"""Extract plain text from uploaded documents -- every common format.

Strategy: fast native parsers for the common text-bearing formats, and
LibreOffice (headless) as a universal fallback for everything in the office
family. Whatever the input, the pipeline only ever sees plain text (with
paragraph breaks, which the chunker relies on).

- text-like (.txt/.md/.csv/.tsv/.json/.xml/.log) : decode
- .html/.htm                                      : strip tags
- .pdf                                            : pypdf
- .docx                                           : python-docx (paragraphs + tables)
- .epub                                           : unzip + strip xhtml
- office family (.doc/.odt/.rtf/.ppt/.pptx/.xls/.xlsx/.odp/.ods) : LibreOffice
"""

from __future__ import annotations

import html as _html
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from io import BytesIO

from .errors import ValidationError

_TEXT_EXT = {".txt", ".md", ".markdown", ".text", ".log", ".csv", ".tsv", ".json", ".xml", ""}
_HTML_EXT = {".html", ".htm", ".xhtml"}
_OFFICE_EXT = {".doc", ".odt", ".rtf", ".ppt", ".pptx", ".xls", ".xlsx", ".odp", ".ods"}
SUPPORTED_EXTENSIONS = _TEXT_EXT | _HTML_EXT | _OFFICE_EXT | {".pdf", ".docx", ".epub"}

_MIME = {
    ".txt": "text/plain", ".md": "text/markdown", ".csv": "text/csv", ".json": "application/json",
    ".html": "text/html", ".htm": "text/html", ".xml": "application/xml",
    ".pdf": "application/pdf", ".epub": "application/epub+zip",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword", ".rtf": "application/rtf",
}


class UnsupportedDocument(ValidationError):
    code = "unsupported_document"


def mime_for(filename: str) -> str:
    return _MIME.get(_ext(filename), "application/octet-stream")


def extract_text(filename: str, data: bytes) -> str:
    ext = _ext(filename)
    if ext in _TEXT_EXT:
        return data.decode("utf-8", errors="replace")
    if ext in _HTML_EXT:
        return _strip_html(data.decode("utf-8", errors="replace"))
    if ext == ".pdf":
        return _from_pdf(data)
    if ext == ".docx":
        return _from_docx(data)
    if ext == ".epub":
        return _from_epub(data)
    if ext in _OFFICE_EXT:
        return _from_libreoffice(data, ext)
    raise UnsupportedDocument(
        f"unsupported file type '{ext}'. Supported: {', '.join(sorted(e for e in SUPPORTED_EXTENSIONS if e))}"
    )


def _ext(filename: str) -> str:
    return os.path.splitext(filename or "")[1].lower()


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"[ \t]*\n\s*\n\s*", "\n\n", _html.unescape(text)).strip()


def _from_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    pages = [(p.extract_text() or "").strip() for p in reader.pages]
    return "\n\n".join(p for p in pages if p)


def _from_docx(data: bytes) -> str:
    from docx import Document as Docx

    doc = Docx(BytesIO(data))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(p for p in parts if p is not None)


def _from_epub(data: bytes) -> str:
    out = []
    with zipfile.ZipFile(BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]
        for name in sorted(names):
            out.append(_strip_html(zf.read(name).decode("utf-8", errors="replace")))
    text = "\n\n".join(t for t in out if t)
    if not text.strip():
        raise UnsupportedDocument("could not extract text from .epub")
    return text


def _from_libreoffice(data: bytes, ext: str) -> str:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise UnsupportedDocument(
            f"reading {ext} requires LibreOffice; convert to .docx/.pdf and re-upload")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, f"input{ext}")
        with open(src, "wb") as fh:
            fh.write(data)
        env = {**os.environ, "HOME": tmp}
        proc = subprocess.run(
            [soffice, "--headless", "--convert-to", "txt:Text", "--outdir", tmp, src],
            timeout=180, capture_output=True, env=env)
        outs = [f for f in os.listdir(tmp) if f.endswith(".txt")]
        if not outs:
            detail = (proc.stderr or proc.stdout).decode("utf-8", "replace")[:200]
            raise UnsupportedDocument(f"could not convert {ext} file: {detail}")
        with open(os.path.join(tmp, outs[0]), encoding="utf-8", errors="replace") as fh:
            return fh.read()
