"""
Ingestion orchestrator — the offline pipeline from the ingestion
design session:

    source -> loader -> chunker -> metadata tagger -> embedder -> upsert

Design decisions this file encodes:
  - Idempotent upsert: each chunk's Pinecone vector ID is
    sha256(source_identifier + chunk_index), so re-ingesting the same
    document overwrites its own vectors instead of duplicating them.
  - embed_model is stamped into every chunk's metadata — this is what
    lets the cache-key design and the "embedding model drift" silent
    failure (S1) both detect a model change after the fact.
  - Chunks are embedded in batches (OpenAI's embeddings.create
    already accepts a list) rather than one API call per chunk —
    same principle as the grader's asyncio.gather, but here it's a
    single batched call, since OpenAI's embeddings endpoint natively
    supports up to ~2048 inputs per request.
  - Returns a structured result (chunks_ingested, chunks_dropped,
    document_id) rather than raising on partial success — a document
    that yields zero usable chunks after the orphan-chunk filter is a
    real, expected outcome (e.g. a title-page-only PDF), not a crash.
"""

from __future__ import annotations

import hashlib
import logging

from app.ingestion.chunker import chunk_text_with_stats
from app.ingestion.loaders import LoaderError, load_pdf, load_text, load_url
from app.llm.embeddings import embed_texts
from app.retrieval.base import VectorStore

logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = "text-embedding-3-small"  # must match app.llm.embeddings.EMBEDDING_MODEL


class IngestionResult:
    def __init__(
        self,
        *,
        title: str,
        source: str,
        chunks_ingested: int,
        chunks_dropped: int,
    ):
        self.title = title
        self.source = source
        self.chunks_ingested = chunks_ingested
        self.chunks_dropped = chunks_dropped

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "source": self.source,
            "chunks_ingested": self.chunks_ingested,
            "chunks_dropped": self.chunks_dropped,
        }


def _chunk_vector_id(source: str, chunk_index: int) -> str:
    """Deterministic ID -> re-ingesting the same source overwrites
    its own old vectors instead of creating duplicates alongside
    them. This is the idempotency guarantee from the ingest design."""
    raw = f"{source}::{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def ingest_source(
    *,
    source_type: str,
    source: str | bytes,
    user_id: str,
    vector_store: VectorStore,
    filename: str | None = None,
) -> IngestionResult:
    """Runs the full ingestion pipeline for one document and upserts
    it into `user_id`'s Pinecone namespace.

    source_type: "pdf" | "url" | "text"
    source: PDF bytes, a URL string, or raw text, matching source_type
    """
    loaded = await _load(source_type, source, filename=filename)
    text, title = loaded["text"], loaded["title"]

    source_identifier = filename or (source if isinstance(source, str) else title)

    raw_chunks, dropped_count = chunk_text_with_stats(text)

    if not raw_chunks:
        logger.warning(
            "Document '%s' produced zero usable chunks after filtering (%d dropped)",
            title, dropped_count,
        )
        return IngestionResult(
            title=title, source=source_identifier, chunks_ingested=0, chunks_dropped=dropped_count
        )

    embeddings = await embed_texts(raw_chunks)

    vectors: list[tuple[str, list[float], dict]] = []
    for i, (chunk_content, embedding) in enumerate(zip(raw_chunks, embeddings)):
        vector_id = _chunk_vector_id(source_identifier, i)
        metadata = {
            "text": chunk_content,
            "source": source_identifier,
            "title": title,
            "chunk_index": i,
            "embed_model": EMBED_MODEL_NAME,
        }
        vectors.append((vector_id, embedding, metadata))

    await vector_store.upsert(vectors=vectors, namespace=user_id)

    return IngestionResult(
        title=title,
        source=source_identifier,
        chunks_ingested=len(vectors),
        chunks_dropped=dropped_count,
    )


async def _load(source_type: str, source: str | bytes, *, filename: str | None) -> dict:
    if source_type == "pdf":
        if not isinstance(source, bytes):
            raise LoaderError("source_type='pdf' requires bytes input")
        return load_pdf(source, filename=filename or "uploaded.pdf")
    if source_type == "url":
        if not isinstance(source, str):
            raise LoaderError("source_type='url' requires a string input")
        return await load_url(source)
    if source_type == "text":
        if not isinstance(source, str):
            raise LoaderError("source_type='text' requires a string input")
        return load_text(source, title=filename or "Pasted text")

    raise LoaderError(f"Unknown source_type: {source_type!r}")
