"""
Chunker — splits loaded document text into overlapping chunks ready
for embedding.

Design decisions from the ingestion pipeline session:
  - 512 tokens per chunk, 64 token overlap — the empirically safe
    default for research papers (large enough for semantic
    coherence, small enough to avoid noisy retrieval).
  - Splits recursively: paragraph breaks first, then sentence breaks,
    then word boundaries — never mid-word.
  - Token counting uses tiktoken (cl100k_base, matching
    text-embedding-3-small), not character count — a naive character
    split systematically over- or under-fills chunks depending on
    content density.
  - Orphan chunk filter (edge case 2.4): chunks under
    MIN_CHUNK_TOKENS are dropped — these are almost always table-of-
    contents fragments or section-heading noise with a deceptively
    high embedding similarity to many queries and zero information.
  - Figure/table reference stripping (silent failure S4): a chunk
    claiming "as shown in Figure 3" is misleading when Figure 3 was
    never ingested (PDFs lose images in text extraction). We replace
    these references rather than deleting them, so the synthesizer's
    system prompt (which is told to flag "[visual reference
    omitted]") has something concrete to react to.
"""

from __future__ import annotations

import logging
import re

import tiktoken

logger = logging.getLogger(__name__)

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
MIN_CHUNK_TOKENS = 80  # edge case 2.4 — orphan chunk filter

_FIGURE_REFERENCE_PATTERN = re.compile(
    r"\b(as shown in|see|refer to)\s+(figure|table|fig\.?)\s*\d+\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------
# Lazy, fault-tolerant tokenizer loading.
#
# tiktoken.get_encoding() downloads its vocab file over the network on
# first use (and caches it locally after that). Doing this at MODULE
# IMPORT TIME is fragile in three real ways: it makes every cold start
# slower, it makes the whole module unimportable in network-restricted
# environments (CI runners, some corporate networks, some sandboxes),
# and it means a transient network blip during deploy can take down
# ingestion entirely.
#
# Instead: load lazily on first real use, cache the result, and if the
# download genuinely fails, fall back to a simple character-based
# approximation (~4 chars/token for English) rather than crashing.
# This is the same "external dependency degrades, never crashes"
# pattern used for Pinecone (edge case: degraded=True) and MLflow
# (observability failures never break the critical path) elsewhere
# in this project.
# --------------------------------------------------------------------------

_encoding: "tiktoken.Encoding | None" = None
_encoding_load_failed = False
_APPROX_CHARS_PER_TOKEN = 4


def _get_encoding() -> "tiktoken.Encoding | None":
    global _encoding, _encoding_load_failed
    if _encoding is not None or _encoding_load_failed:
        return _encoding
    try:
        _encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as e:  # network error, offline, blocked domain, etc.
        logger.warning(
            "tiktoken encoding unavailable (%s) — falling back to approximate "
            "token counting (~%d chars/token). Chunk boundaries will be less "
            "precise but ingestion will still function.",
            e,
            _APPROX_CHARS_PER_TOKEN,
        )
        _encoding_load_failed = True
    return _encoding


def count_tokens(text: str) -> int:
    encoding = _get_encoding()
    if encoding is not None:
        return len(encoding.encode(text))
    if not text:
        return 0
    return max(1, len(text) // _APPROX_CHARS_PER_TOKEN)


def _strip_figure_references(text: str) -> str:
    """Edge case S4: replaces dangling visual references with an
    explicit marker instead of leaving a claim that depends on
    content we don't have."""
    return _FIGURE_REFERENCE_PATTERN.sub("[visual reference omitted]", text)


def _split_on_pattern(text: str, pattern: str) -> list[str]:
    parts = re.split(pattern, text)
    return [p for p in parts if p.strip()]


def _recursive_split(text: str, max_tokens: int) -> list[str]:
    """Splits `text` into pieces each under `max_tokens`, trying
    paragraph breaks first, then sentences, then words — never
    splitting mid-word."""
    if count_tokens(text) <= max_tokens:
        return [text]

    for pattern in (r"\n\s*\n", r"(?<=[.!?])\s+", r"\s+"):
        pieces = _split_on_pattern(text, pattern)
        if len(pieces) > 1:
            break
    else:
        # Single unsplittable token-dense blob — hard cut as last resort
        encoding = _get_encoding()
        if encoding is not None:
            tokens = encoding.encode(text)
            return [encoding.decode(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
        # Approximate fallback: cut by character count instead of token
        # count. Less precise chunk boundaries, but still functional.
        approx_chars = max_tokens * _APPROX_CHARS_PER_TOKEN
        return [text[i : i + approx_chars] for i in range(0, len(text), approx_chars)]

    result: list[str] = []
    buffer = ""
    for piece in pieces:
        candidate = f"{buffer} {piece}".strip() if buffer else piece
        if count_tokens(candidate) <= max_tokens:
            buffer = candidate
        else:
            if buffer:
                result.append(buffer)
            if count_tokens(piece) > max_tokens:
                result.extend(_recursive_split(piece, max_tokens))
                buffer = ""
            else:
                buffer = piece
    if buffer:
        result.append(buffer)
    return result


def _add_overlap(chunks: list[str], overlap_tokens: int) -> list[str]:
    """Prepends the tail of the previous chunk to each subsequent
    chunk, so context isn't lost at chunk boundaries (e.g. a sentence
    that got cut in half)."""
    if len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        encoding = _get_encoding()
        if encoding is not None:
            prev_tokens = encoding.encode(chunks[i - 1])
            tail = (
                encoding.decode(prev_tokens[-overlap_tokens:])
                if len(prev_tokens) > overlap_tokens
                else chunks[i - 1]
            )
        else:
            approx_chars = overlap_tokens * _APPROX_CHARS_PER_TOKEN
            tail = chunks[i - 1][-approx_chars:]
        overlapped.append(f"{tail} {chunks[i]}".strip())
    return overlapped


def chunk_text(text: str) -> list[str]:
    """Full pipeline: strip figure references, recursively split to
    CHUNK_SIZE_TOKENS, add CHUNK_OVERLAP_TOKENS of overlap, drop
    anything under MIN_CHUNK_TOKENS."""
    kept, _dropped_count = chunk_text_with_stats(text)
    return kept


def chunk_text_with_stats(text: str) -> tuple[list[str], int]:
    """Same pipeline as chunk_text(), but also reports how many
    chunks were dropped by the orphan-chunk filter (edge case 2.4) —
    used by the ingestion orchestrator to report accurate
    chunks_dropped counts instead of silently discarding that
    information, which is what happened before this function existed
    (IngestionResult.chunks_dropped was hardcoded to 0)."""
    cleaned = _strip_figure_references(text)
    raw_chunks = _recursive_split(cleaned, CHUNK_SIZE_TOKENS)
    overlapped_chunks = _add_overlap(raw_chunks, CHUNK_OVERLAP_TOKENS)

    kept = [c for c in overlapped_chunks if count_tokens(c) >= MIN_CHUNK_TOKENS]
    dropped_count = len(overlapped_chunks) - len(kept)
    return kept, dropped_count
