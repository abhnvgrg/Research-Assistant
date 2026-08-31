from __future__ import annotations

import logging
import re

import tiktoken

logger = logging.getLogger(__name__)

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
MIN_CHUNK_TOKENS = 80

_FIGURE_REFERENCE_PATTERN = re.compile(
    r"\b(as shown in|see|refer to)\s+(figure|table|fig\.?)\s*\d+\b",
    re.IGNORECASE,
)


_encoding: "tiktoken.Encoding | None" = None
_encoding_load_failed = False
_APPROX_CHARS_PER_TOKEN = 4


def _get_encoding() -> "tiktoken.Encoding | None":
    global _encoding, _encoding_load_failed
    if _encoding is not None or _encoding_load_failed:
        return _encoding
    try:
        _encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
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
    return _FIGURE_REFERENCE_PATTERN.sub("[visual reference omitted]", text)


def _split_on_pattern(text: str, pattern: str) -> list[str]:
    parts = re.split(pattern, text)
    return [p for p in parts if p.strip()]


def _recursive_split(text: str, max_tokens: int) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]

    for pattern in (r"\n\s*\n", r"(?<=[.!?])\s+", r"\s+"):
        pieces = _split_on_pattern(text, pattern)
        if len(pieces) > 1:
            break
    else:
        encoding = _get_encoding()
        if encoding is not None:
            tokens = encoding.encode(text)
            return [encoding.decode(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
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
    kept, _dropped_count = chunk_text_with_stats(text)
    return kept


def chunk_text_with_stats(text: str) -> tuple[list[str], int]:
    cleaned = _strip_figure_references(text)
    raw_chunks = _recursive_split(cleaned, CHUNK_SIZE_TOKENS)
    overlapped_chunks = _add_overlap(raw_chunks, CHUNK_OVERLAP_TOKENS)

    kept = [c for c in overlapped_chunks if count_tokens(c) >= MIN_CHUNK_TOKENS]
    dropped_count = len(overlapped_chunks) - len(kept)
    return kept, dropped_count
