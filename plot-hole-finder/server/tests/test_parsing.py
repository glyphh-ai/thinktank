"""Document parsing: txt, pdf, docx (legal-style uploads)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parsing import UnsupportedDocument, extract_text, mime_for

FIXTURES = Path(__file__).parent / "fixtures"


def test_plain_text():
    assert extract_text("notes.txt", b"Hello world") == "Hello world"


def test_pdf_extraction():
    text = extract_text("sample.pdf", (FIXTURES / "sample.pdf").read_bytes())
    assert "green" in text and "blue" in text


def test_docx_extraction_includes_tables():
    text = extract_text("sample.docx", (FIXTURES / "sample.docx").read_bytes())
    assert "green" in text and "blue" in text
    assert "Exhibit A" in text  # table cells are pulled (legal docs use tables)


def test_html_extraction():
    html = b"<html><head><style>x{}</style></head><body><h1>Title</h1>" \
           b"<p>Elena's eyes were green.</p></body></html>"
    text = extract_text("page.html", html)
    assert "Elena's eyes were green." in text
    assert "<" not in text and "style" not in text


def test_csv_is_text():
    assert "alice" in extract_text("data.csv", b"name,age\nalice,30")


def test_office_via_libreoffice():
    import shutil

    if not (shutil.which("soffice") or shutil.which("libreoffice")):
        pytest.skip("LibreOffice not installed")
    rtf = rb"{\rtf1\ansi\deff0{\fonttbl{\f0 Arial;}}Marcus had black hair.\par}"
    try:
        text = extract_text("note.rtf", rtf)
    except UnsupportedDocument:
        pytest.skip("LibreOffice present but non-functional in this environment")
    assert "black hair" in text.lower()


def test_unsupported_extension():
    with pytest.raises(UnsupportedDocument):
        extract_text("archive.zip", b"PK\x03\x04")


def test_mime_for():
    assert mime_for("a.pdf") == "application/pdf"
    assert mime_for("a.docx").endswith("wordprocessingml.document")
