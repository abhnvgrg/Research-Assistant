"""
Document loader — normalizes any supported source (PDF bytes, a
webpage URL, or raw text) into a single (text, title) tuple that the
chunker consumes next.

Design decisions from the ingestion pipeline session:
  - PDF text extraction uses pypdf (pure Python, no system
    dependencies like poppler — important for a simple Docker image).
  - URL loading strips script/style tags and boilerplate before
    returning text — a raw HTML dump would poison every downstream
    chunk with markup noise.
  - Every loader function has the same signature shape: takes the
    raw source, returns dict(text=str, title=str) — the chunker
    never needs to know which loader produced its input.
"""

from __future__ import annotations

import io
import logging
import re

import httpx
from pypdf import PdfReader

logger = logging.getLogger(__name__)

MAX_PDF_SIZE_BYTES = 20 * 1024 * 1024  # 20MB — matches the DoS-guard design
MAX_URL_FETCH_TIMEOUT = 15.0


class LoaderError(Exception):
    """Raised when a source can't be loaded at all — caught by the
    ingestion orchestrator and reported back as a failed job status,
    never left to crash the background worker."""


def load_pdf(file_bytes: bytes, *, filename: str = "uploaded.pdf") -> dict:
    """Extracts text from a PDF's bytes. Rejects oversized files
    before doing any parsing work (edge case: 500MB PDF DoS)."""
    if len(file_bytes) > MAX_PDF_SIZE_BYTES:
        raise LoaderError(
            f"PDF exceeds {MAX_PDF_SIZE_BYTES // (1024 * 1024)}MB limit "
            f"({len(file_bytes) / (1024 * 1024):.1f}MB received)"
        )

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        raise LoaderError(f"Could not open PDF: {e}") from e

    if reader.is_encrypted:
        raise LoaderError("PDF is password-protected — cannot extract text")

    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as e:  # a single malformed page shouldn't kill the whole doc
            logger.warning("Failed to extract a page from %s: %s", filename, e)

    full_text = "\n\n".join(p for p in pages_text if p.strip())

    if not full_text.strip():
        raise LoaderError(
            "No extractable text found — this PDF may be scanned images "
            "without OCR, which this loader does not support"
        )

    title = _guess_title_from_text(full_text) or filename
    return {"text": full_text, "title": title}


async def load_url(url: str) -> dict:
    """Fetches a webpage and extracts readable text, stripping
    script/style/nav boilerplate."""
    try:
        async with httpx.AsyncClient(timeout=MAX_URL_FETCH_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(url, headers={"User-Agent": "ResearchAssistant/1.0"})
            response.raise_for_status()
    except httpx.HTTPError as e:
        raise LoaderError(f"Could not fetch URL: {e}") from e

    html = response.text
    text = _strip_html_to_text(html)

    if not text.strip():
        raise LoaderError("No readable text extracted from this URL")

    title = _extract_html_title(html) or url
    return {"text": text, "title": title}


def load_text(raw_text: str, *, title: str = "Pasted text") -> dict:
    """Direct text input — no extraction needed, but still validated
    for emptiness so downstream chunking never runs on nothing."""
    if not raw_text.strip():
        raise LoaderError("Text input is empty")
    return {"text": raw_text, "title": title}


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _strip_html_to_text(html: str) -> str:
    # Remove script/style blocks entirely (including their content)
    html = re.sub(r"<(script|style|nav|footer|header)\b[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse excessive whitespace left behind
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _extract_html_title(html: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def _guess_title_from_text(text: str) -> str | None:
    """Best-effort: the first non-empty line of a PDF's extracted
    text is often its title. Falls back to None (caller uses
    filename instead) if that line looks unreasonable."""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), None)
    if first_line and 3 <= len(first_line) <= 200:
        return first_line
    return None
