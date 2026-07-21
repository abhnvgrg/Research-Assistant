"""
Loader tests — covers the three source types (PDF, URL, text) and
every edge case the loaders design session called out explicitly:
  - oversized PDF rejected before parsing (DoS guard, edge case 6.6)
  - password-protected PDF rejected with a clear error, not a crash
  - a single malformed page doesn't kill extraction of the rest
  - URL fetches never touch the real network — mocked via
    httpx.MockTransport, consistent with the "unit tests never hit
    external services" rule
  - HTML boilerplate (script/style/nav/footer) is stripped, not just
    the visible tags
"""

from __future__ import annotations

import io

import httpx
import pytest
from reportlab.pdfgen import canvas

import app.ingestion.loaders as loaders_module
from app.ingestion.loaders import LoaderError, load_pdf, load_text, load_url


def _make_pdf_bytes(lines: list[str]) -> bytes:
    """Builds a real, single-page, parseable PDF with the given lines
    of text — using reportlab to generate genuine PDF structure
    rather than a hand-rolled fixture, so pypdf's real extraction
    code path runs."""
    return _make_multipage_pdf_bytes([lines])


def _make_multipage_pdf_bytes(pages: list[list[str]]) -> bytes:
    """Like _make_pdf_bytes, but each inner list becomes its own
    genuine PDF page (calls canvas.showPage() between pages) —
    needed for tests that must exercise pypdf's per-page extraction
    loop across more than one real page."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for page_lines in pages:
        y = 750
        for line in page_lines:
            c.drawString(100, y, line)
            y -= 30
        c.showPage()
    c.save()
    return buf.getvalue()


def _make_encrypted_pdf_bytes(password: str = "secret") -> bytes:
    from pypdf import PdfReader, PdfWriter

    plain = _make_pdf_bytes(["Confidential content"])
    reader = PdfReader(io.BytesIO(plain))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(password)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


# ---- load_pdf ----

def test_load_pdf_extracts_text_and_guesses_title():
    pdf_bytes = _make_pdf_bytes(["Attention Is All You Need", "We propose the Transformer."])
    result = load_pdf(pdf_bytes, filename="paper.pdf")

    assert "Attention Is All You Need" in result["text"]
    assert "Transformer" in result["text"]
    assert result["title"] == "Attention Is All You Need"


def test_load_pdf_falls_back_to_filename_when_title_guess_fails():
    """A first line that's implausible as a title (too short, e.g.
    just a page number) should fall back to the filename."""
    pdf_bytes = _make_pdf_bytes(["1"])  # 1 char, below the 3-char minimum
    result = load_pdf(pdf_bytes, filename="report.pdf")
    assert result["title"] == "report.pdf"


def test_load_pdf_rejects_oversized_files_without_parsing():
    """Edge case: DoS guard. An oversized 'PDF' (garbage bytes are
    fine here — the size check must happen BEFORE any parsing is
    attempted, so this must reject without ever touching PdfReader)."""
    oversized = b"x" * (21 * 1024 * 1024)  # 21MB, over the 20MB limit

    with pytest.raises(LoaderError, match="exceeds"):
        load_pdf(oversized, filename="huge.pdf")


def test_load_pdf_rejects_password_protected_pdfs():
    encrypted_bytes = _make_encrypted_pdf_bytes()

    with pytest.raises(LoaderError, match="password-protected"):
        load_pdf(encrypted_bytes, filename="locked.pdf")


def test_load_pdf_rejects_corrupt_bytes():
    with pytest.raises(LoaderError, match="Could not open PDF"):
        load_pdf(b"this is not a pdf at all", filename="fake.pdf")


def test_load_pdf_survives_a_single_malformed_page(monkeypatch):
    """A page that throws during extract_text() must not take down
    extraction of the other pages in the same document."""
    pdf_bytes = _make_multipage_pdf_bytes([["Good page one"], ["Good page two"]])

    import pypdf

    original_extract = pypdf.PageObject.extract_text
    call_count = {"n": 0}

    def flaky_extract(self, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ValueError("simulated malformed page")
        return original_extract(self, *args, **kwargs)

    monkeypatch.setattr(pypdf.PageObject, "extract_text", flaky_extract)

    result = load_pdf(pdf_bytes, filename="mixed.pdf")
    assert "Good page two" in result["text"]


def test_load_pdf_rejects_pdf_with_no_extractable_text():
    """Simulates a scanned-image PDF with no OCR — pypdf successfully
    opens it but extracts nothing."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)  # no text content at all
    buf = io.BytesIO()
    writer.write(buf)

    with pytest.raises(LoaderError, match="No extractable text"):
        load_pdf(buf.getvalue(), filename="scanned.pdf")


# ---- load_url ----

def _patch_http_response(monkeypatch, *, status_code: int = 200, text: str = "", raise_error: bool = False):
    async def handler(request):
        if raise_error:
            raise httpx.ConnectError("simulated connection failure")
        return httpx.Response(status_code, text=text)

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs.pop("transport", None)
            super().__init__(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(loaders_module.httpx, "AsyncClient", PatchedAsyncClient)


async def test_load_url_extracts_text_and_title(monkeypatch):
    html = (
        "<html><head><title>Understanding Attention</title></head>"
        "<body><script>var x = 1;</script><p>Attention lets models focus.</p></body></html>"
    )
    _patch_http_response(monkeypatch, text=html)

    result = await load_url("https://example.com/article")

    assert result["title"] == "Understanding Attention"
    assert "Attention lets models focus." in result["text"]
    assert "var x = 1" not in result["text"]  # script content stripped


async def test_load_url_strips_nav_and_footer_boilerplate(monkeypatch):
    html = (
        "<html><body>"
        "<nav>Home About Contact</nav>"
        "<p>The actual article content goes here.</p>"
        "<footer>Copyright 2026</footer>"
        "</body></html>"
    )
    _patch_http_response(monkeypatch, text=html)

    result = await load_url("https://example.com")

    assert "The actual article content goes here." in result["text"]
    assert "Home About Contact" not in result["text"]
    assert "Copyright 2026" not in result["text"]


async def test_load_url_falls_back_to_url_when_no_title_tag(monkeypatch):
    _patch_http_response(monkeypatch, text="<html><body><p>No title here.</p></body></html>")

    result = await load_url("https://example.com/no-title-page")

    assert result["title"] == "https://example.com/no-title-page"


async def test_load_url_raises_loader_error_on_http_failure(monkeypatch):
    _patch_http_response(monkeypatch, status_code=404, text="Not Found")

    with pytest.raises(LoaderError, match="Could not fetch URL"):
        await load_url("https://example.com/missing")


async def test_load_url_raises_loader_error_on_connection_failure(monkeypatch):
    _patch_http_response(monkeypatch, raise_error=True)

    with pytest.raises(LoaderError, match="Could not fetch URL"):
        await load_url("https://unreachable.example.com")


async def test_load_url_rejects_empty_extracted_text(monkeypatch):
    _patch_http_response(monkeypatch, text="<html><body><script>only script, no text</script></body></html>")

    with pytest.raises(LoaderError, match="No readable text"):
        await load_url("https://example.com/empty")


# ---- load_text ----

def test_load_text_returns_input_unchanged():
    result = load_text("Some pasted research notes.", title="My Notes")
    assert result == {"text": "Some pasted research notes.", "title": "My Notes"}


def test_load_text_uses_default_title_when_not_provided():
    result = load_text("Some content.")
    assert result["title"] == "Pasted text"


def test_load_text_rejects_empty_input():
    with pytest.raises(LoaderError, match="empty"):
        load_text("   \n  ")
