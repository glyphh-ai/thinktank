"""Extract plain text from uploaded documents.

Legal/manuscript uploads arrive as raw text, PDF, or Word. We convert everything
to plain text (preserving paragraph breaks, which the chunker relies on) before
the pipeline ever sees it. The original bytes are kept in blob storage; this
module produces the text representation the analysis runs against.

- .txt/.md/.text : decode
- .pdf           : pypdf
- .docx          : python-docx (paragraphs + table cells -- legal docs use tables)
- .doc (legacy)  : best-effort via LibreOffice headless, if available
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from io import BytesIO

from .errors import ValidationError

SUPPORTED_EXTENSIONS = {".txt", ".md", ".text", ".pdf", ".docx", ".doc"}

_MIME_BY_EXT = {
    ".txt": "text/plain", ".md": "text/markdown", ".text": "text/plain",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
}


class UnsupportedDocument(ValidationError):
    code = "unsupported_document"


def mime_for(filename: str) -> str:
    return _MIME_BY_EXT.get(_ext(filename), "application/octet-stream")


def extract_text(filename: str, data: bytes) -> str:
    """Convert an uploaded document to plain text by file extension."""
    ext = _ext(filename)
    if ext in {".txt", ".md", ".text", ""}:
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        return _from_pdf(data)
    if ext == ".docx":
        return _from_docx(data)
    if ext == ".doc":
        return _from_doc(data)
    raise UnsupportedDocument(
        f"unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def _ext(filename: str) -> str:
    return os.path.splitext(filename or "")[1].lower()


def _from_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)


def _from_docx(data: bytes) -> str:
    from docx import Document as Docx

    doc = Docx(BytesIO(data))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(part for part in parts if part is not None)


def _from_doc(data: bytes) -> str:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise UnsupportedDocument(
            "legacy .doc requires LibreOffice; convert to .docx or .pdf and re-upload"
        )
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "input.doc")
        with open(src, "wb") as fh:
            fh.write(data)
        # LibreOffice needs a writable HOME for its profile.
        env = {**os.environ, "HOME": tmp}
        subprocess.run(
            [soffice, "--headless", "--convert-to", "txt:Text", "--outdir", tmp, src],
            check=True, timeout=120, capture_output=True, env=env,
        )
        out = os.path.join(tmp, "input.txt")
        if not os.path.exists(out):
            raise UnsupportedDocument("could not convert .doc file")
        with open(out, encoding="utf-8", errors="replace") as fh:
            return fh.read()
