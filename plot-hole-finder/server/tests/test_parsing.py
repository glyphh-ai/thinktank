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


def test_unsupported_extension():
    with pytest.raises(UnsupportedDocument):
        extract_text("archive.zip", b"PK\x03\x04")


def test_mime_for():
    assert mime_for("a.pdf") == "application/pdf"
    assert mime_for("a.docx").endswith("wordprocessingml.document")
