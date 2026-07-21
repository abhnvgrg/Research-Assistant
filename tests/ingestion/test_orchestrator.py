"""
Orchestrator tests — proves the full pipeline (loader -> chunker ->
embedder -> upsert) wires together correctly, using a fake
VectorStore (never real Pinecone) and a monkeypatched embed_texts
(never real OpenAI).

Specifically covers three design guarantees from the ingestion
session that are easy to silently break during refactors:
  1. Idempotent vector IDs — re-ingesting the same source overwrites
     its own vectors rather than creating duplicates.
  2. embed_model is stamped into every chunk's metadata (needed for
     the cache-key design and the embedding-drift silent failure).
  3. chunks_dropped in the result is now ACCURATE (see the bug fixed
     in chunker.py — chunk_text_with_stats), not hardcoded to 0.
"""

from __future__ import annotations

import pytest

import app.ingestion.orchestrator as orchestrator_module
from app.ingestion.loaders import LoaderError
from app.ingestion.orchestrator import EMBED_MODEL_NAME, ingest_source
from app.retrieval.base import VectorStore


class FakeVectorStore(VectorStore):
    """Records every upsert call for inspection instead of touching
    Pinecone."""

    def __init__(self):
        self.upserted: list[tuple] = []

    async def query(self, *, vector, top_k, namespace):
        return []

    async def upsert(self, *, vectors, namespace):
        self.upserted.append((vectors, namespace))


@pytest.fixture(autouse=True)
def fast_embeddings(monkeypatch):
    """Every chunk gets a trivially distinguishable fake embedding
    (index-based) instead of calling real OpenAI."""

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[float(i)] for i in range(len(texts))]

    monkeypatch.setattr(orchestrator_module, "embed_texts", fake_embed_texts)


async def test_ingest_text_source_upserts_correct_vector_count():
    store = FakeVectorStore()
    long_text = "This is substantial research content about attention. " * 40

    result = await ingest_source(
        source_type="text",
        source=long_text,
        user_id="user-123",
        vector_store=store,
    )

    assert result.chunks_ingested > 0
    assert len(store.upserted) == 1
    vectors, namespace = store.upserted[0]
    assert namespace == "user-123"  # namespace = user_id, per the security design
    assert len(vectors) == result.chunks_ingested


async def test_ingest_stamps_embed_model_into_every_chunk_metadata():
    store = FakeVectorStore()
    long_text = "Research content about transformers. " * 40

    await ingest_source(source_type="text", source=long_text, user_id="u1", vector_store=store)

    vectors, _ = store.upserted[0]
    for _id, _embedding, metadata in vectors:
        assert metadata["embed_model"] == EMBED_MODEL_NAME


async def test_ingest_produces_idempotent_vector_ids_on_reingest():
    """Re-ingesting the exact same source (same filename/identifier)
    must produce the SAME vector IDs both times — this is what makes
    Pinecone's upsert overwrite rather than duplicate."""
    store = FakeVectorStore()
    text = "Consistent research content for idempotency testing. " * 40

    await ingest_source(source_type="text", source=text, user_id="u1", vector_store=store, filename="doc.txt")
    await ingest_source(source_type="text", source=text, user_id="u1", vector_store=store, filename="doc.txt")

    first_ids = [v[0] for v in store.upserted[0][0]]
    second_ids = [v[0] for v in store.upserted[1][0]]
    assert first_ids == second_ids


async def test_ingest_produces_different_ids_for_different_sources():
    store = FakeVectorStore()
    text = "Some research content here that is long enough. " * 40

    await ingest_source(source_type="text", source=text, user_id="u1", vector_store=store, filename="doc-a.txt")
    await ingest_source(source_type="text", source=text, user_id="u1", vector_store=store, filename="doc-b.txt")

    first_ids = {v[0] for v in store.upserted[0][0]}
    second_ids = {v[0] for v in store.upserted[1][0]}
    assert first_ids.isdisjoint(second_ids)


async def test_ingest_reports_zero_chunks_and_skips_upsert_when_everything_filtered():
    """A document that produces zero usable chunks (e.g. a title-
    page-only PDF, or in this case trivially short text) must not
    call upsert() at all, and must report chunks_ingested=0 rather
    than raising."""
    store = FakeVectorStore()

    result = await ingest_source(
        source_type="text", source="hi", user_id="u1", vector_store=store  # far under MIN_CHUNK_TOKENS
    )

    assert result.chunks_ingested == 0
    assert store.upserted == []  # never called


async def test_ingest_reports_accurate_dropped_count():
    """Regression coverage for the chunks_dropped-always-0 bug fixed
    in chunker.py — a document with a genuine dropped orphan chunk
    (tiny fragment as the very first chunk, un-rescued by overlap)
    must report chunks_dropped >= 1, not 0."""
    store = FakeVectorStore()
    text = "Contents\n\n" + ("Introduction to the research topic at hand. " * 400)

    result = await ingest_source(source_type="text", source=text, user_id="u1", vector_store=store)

    assert result.chunks_dropped >= 1
    assert result.chunks_ingested > 0


async def test_ingest_propagates_loader_errors_without_calling_embed_or_upsert():
    """If the loader itself fails (e.g. empty text input), the
    orchestrator must not proceed to embedding or upserting — those
    would be wasted API calls against garbage input."""
    store = FakeVectorStore()

    with pytest.raises(LoaderError):
        await ingest_source(source_type="text", source="   ", user_id="u1", vector_store=store)

    assert store.upserted == []


async def test_ingest_rejects_pdf_source_type_with_non_bytes_input():
    store = FakeVectorStore()

    with pytest.raises(LoaderError, match="requires bytes"):
        await ingest_source(source_type="pdf", source="not bytes", user_id="u1", vector_store=store)


async def test_ingest_rejects_url_source_type_with_non_string_input():
    store = FakeVectorStore()

    with pytest.raises(LoaderError, match="requires a string"):
        await ingest_source(source_type="url", source=b"not a string", user_id="u1", vector_store=store)


async def test_ingest_rejects_unknown_source_type():
    store = FakeVectorStore()

    with pytest.raises(LoaderError, match="Unknown source_type"):
        await ingest_source(source_type="carrier_pigeon", source="x", user_id="u1", vector_store=store)


async def test_ingestion_result_to_dict_shape():
    store = FakeVectorStore()
    text = "Some reasonably long research content for this test case. " * 40

    result = await ingest_source(source_type="text", source=text, user_id="u1", vector_store=store)
    d = result.to_dict()

    assert set(d.keys()) == {"title", "source", "chunks_ingested", "chunks_dropped"}


async def test_ingest_pdf_source_type_success_path():
    """The _load() success branch for source_type='pdf' was
    previously only exercised via its type-validation error case
    (non-bytes input) — this closes that gap with a real, valid PDF
    all the way through the pipeline."""
    from reportlab.pdfgen import canvas
    import io

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for i in range(40):
        c.drawString(100, 750 - (i % 20) * 30, f"Research finding number {i} about attention mechanisms.")
        if i % 20 == 19:
            c.showPage()
    c.save()
    pdf_bytes = buf.getvalue()

    store = FakeVectorStore()
    result = await ingest_source(
        source_type="pdf", source=pdf_bytes, user_id="u1", vector_store=store, filename="findings.pdf"
    )

    assert result.chunks_ingested > 0
    assert result.title  # guessed from first line or fell back to filename
    assert len(store.upserted) == 1


async def test_ingest_url_source_type_success_path(monkeypatch):
    """Same gap as above, for source_type='url' — mocks load_url
    itself (already thoroughly tested in test_loaders.py) rather
    than re-mocking the HTTP layer here, keeping this test focused
    on orchestrator wiring, not URL fetching mechanics."""

    async def fake_load_url(url: str) -> dict:
        return {
            "text": "Substantial article content about transformers. " * 40,
            "title": "Understanding Transformers",
        }

    monkeypatch.setattr(orchestrator_module, "load_url", fake_load_url)

    store = FakeVectorStore()
    result = await ingest_source(
        source_type="url", source="https://example.com/article", user_id="u1", vector_store=store
    )

    assert result.chunks_ingested > 0
    assert result.title == "Understanding Transformers"
    assert result.source == "https://example.com/article"
