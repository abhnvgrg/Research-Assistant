from __future__ import annotations

import pytest

import app.ingestion.orchestrator as orchestrator_module
from app.ingestion.loaders import LoaderError
from app.ingestion.orchestrator import EMBED_MODEL_NAME, ingest_source
from app.retrieval.base import VectorStore


class FakeVectorStore(VectorStore):
    def __init__(self):
        self.upserted: list[tuple] = []

    async def query(self, *, vector, top_k, namespace):
        return []

    async def upsert(self, *, vectors, namespace):
        self.upserted.append((vectors, namespace))


@pytest.fixture(autouse=True)
def fast_embeddings(monkeypatch):
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
    assert namespace == "user-123"
    assert len(vectors) == result.chunks_ingested


async def test_ingest_stamps_embed_model_into_every_chunk_metadata():
    store = FakeVectorStore()
    long_text = "Research content about transformers. " * 40

    await ingest_source(source_type="text", source=long_text, user_id="u1", vector_store=store)

    vectors, _ = store.upserted[0]
    for _id, _embedding, metadata in vectors:
        assert metadata["embed_model"] == EMBED_MODEL_NAME


async def test_ingest_produces_idempotent_vector_ids_on_reingest():
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
    store = FakeVectorStore()

    result = await ingest_source(
        source_type="text", source="hi", user_id="u1", vector_store=store
    )

    assert result.chunks_ingested == 0
    assert store.upserted == []


async def test_ingest_reports_accurate_dropped_count():
    store = FakeVectorStore()
    text = "Contents\n\n" + ("Introduction to the research topic at hand. " * 400)

    result = await ingest_source(source_type="text", source=text, user_id="u1", vector_store=store)

    assert result.chunks_dropped >= 1
    assert result.chunks_ingested > 0


async def test_ingest_propagates_loader_errors_without_calling_embed_or_upsert():
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
    assert result.title
    assert len(store.upserted) == 1


async def test_ingest_url_source_type_success_path(monkeypatch):
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
