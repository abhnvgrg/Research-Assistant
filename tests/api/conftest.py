"""
Shared fixtures for API-layer tests.

`stub_all_graph_calls` patches every external `_call_*` function in
app.graph.nodes at once — the API tests care about routing, auth,
SSE framing, and the store, not about re-proving node-level behavior
already covered exhaustively in tests/test_*_node.py. One shared
happy-path stub keeps these tests focused on what they're actually
testing.

`fresh_run_store` resets the module-level RunStore singleton before
each test so runs from one test can never leak into another — the
store is process-global by design (mirroring a real deployment's
single shared store), so tests must explicitly isolate themselves.

`make_bearer_header()` mints a REAL, correctly-signed HS256 JWT for
tests — now that app/api/deps.py performs genuine cryptographic
verification (see the auth-hardening fix), an arbitrary string like
"Bearer demo-token-abc" is correctly rejected as an invalid
signature. SUPABASE_JWT_SECRET is set to a fixed test-only value at
import time so every test in this package (and downstream imports of
this helper from other test files) can mint tokens without needing
per-test setup.
"""

from __future__ import annotations

import os
import time

import jwt
import pytest
from httpx import ASGITransport, AsyncClient

import app.graph.nodes as nodes
from app.main import app
from app.store import RunStore
import app.store as store_module
from app.ingest_store import JobStore

TEST_JWT_SECRET = "test-only-secret-never-used-for-anything-real"
os.environ["SUPABASE_JWT_SECRET"] = TEST_JWT_SECRET


def make_bearer_header(user_id: str, *, expired: bool = False) -> dict[str, str]:
    """Mints a real HS256 JWT with the same shape a genuine
    Supabase-issued token has (sub, aud, exp) and returns it as an
    Authorization header dict, ready to pass to client.get/post."""
    now = int(time.time())
    payload = {
        "sub": user_id,
        "aud": "authenticated",
        "exp": now - 60 if expired else now + 3600,
    }
    token = jwt.encode(payload, TEST_JWT_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def fresh_run_store(monkeypatch):
    """Every test gets its own RunStore instance. Patched via the
    get_run_store() GETTER now (routes.py/sse.py/runner.py call
    `from app.store import get_run_store` and invoke it at call time,
    not `from app.store import run_store` directly) — same pattern as
    fresh_quota_store, needed so get_run_store() can transparently
    return a real Supabase-backed store once app.db's pool is
    available without every caller needing to know which backend is
    active."""
    fresh = RunStore()
    monkeypatch.setattr(store_module, "run_store", fresh)
    monkeypatch.setattr(store_module, "get_run_store", lambda: fresh)

    import app.api.routes as routes_module
    import app.api.sse as sse_module
    import app.graph.runner as runner_module

    monkeypatch.setattr(routes_module, "get_run_store", lambda: fresh)
    monkeypatch.setattr(sse_module, "get_run_store", lambda: fresh)
    monkeypatch.setattr(runner_module, "get_run_store", lambda: fresh)

    return fresh


@pytest.fixture(autouse=True)
def fresh_job_store(monkeypatch):
    """Same isolation guarantee as fresh_run_store, for the
    ingestion job store — patched via get_job_store() now."""
    fresh = JobStore()

    import app.ingest_store as ingest_store_module
    import app.api.routes as routes_module
    import app.ingestion.runner as ingestion_runner_module

    monkeypatch.setattr(ingest_store_module, "job_store", fresh)
    monkeypatch.setattr(ingest_store_module, "get_job_store", lambda: fresh)
    monkeypatch.setattr(routes_module, "get_job_store", lambda: fresh)
    monkeypatch.setattr(ingestion_runner_module, "get_job_store", lambda: fresh)

    return fresh


@pytest.fixture
def stub_ingest_source(monkeypatch):
    """Stubs ingest_source() itself for route-level tests — ingestion
    logic (chunking, embedding, idempotent IDs, dropped-chunk
    reporting) is already exhaustively covered in
    tests/ingestion/test_orchestrator.py; these route tests only need
    to prove the HTTP layer, background-task wiring, and JobStore
    integration work correctly."""
    import app.ingestion.runner as ingestion_runner_module
    from app.ingestion.orchestrator import IngestionResult

    async def fake_ingest_source(*, source_type, source, user_id, vector_store, filename=None):
        return IngestionResult(
            title=filename or "Stub Title",
            source=filename or (source if isinstance(source, str) else "stub-source"),
            chunks_ingested=3,
            chunks_dropped=0,
        )

    monkeypatch.setattr(ingestion_runner_module, "ingest_source", fake_ingest_source)
    return fake_ingest_source


@pytest.fixture
def stub_all_graph_calls(monkeypatch):
    async def fake_decomposer(query: str) -> tuple[dict, int]:
        return {
            "topic_identified": True,
            "multi_intent": False,
            "recency_required": False,
            "corrected_query": query,
            "sub_questions": [{"question": "sub-q", "intent": "general", "aliases": []}],
        }, 50

    async def fake_vector(sub_qs, user_id):
        return [{"text": "v-chunk", "source": "pinecone://x", "title": "t", "score": 0.8, "origin": "vector"}]

    async def fake_web(sub_qs):
        return [{"text": "w-chunk", "source": "https://x.com", "title": "t", "score": 0.7, "origin": "web"}]

    async def fake_grade(chunk, query):
        return {"relevant": True, "score": 0.9, "reason": "stub"}, 20

    async def fake_synth(query, graded):
        return {
            "answer": "stub answer",
            "citations": {str(i + 1): c["source"] for i, c in enumerate(graded)},
            "tokens_used": 500,
        }

    async def fake_reflect(query, sub_qs, answer):
        return {"pass_": True, "checklist": {}, "gap": "nothing"}, 30

    monkeypatch.setattr(nodes, "_call_decomposer_llm", fake_decomposer)
    monkeypatch.setattr(nodes, "_call_vector_retriever", fake_vector)
    monkeypatch.setattr(nodes, "_call_web_search", fake_web)
    monkeypatch.setattr(nodes, "_call_grader_llm", fake_grade)
    monkeypatch.setattr(nodes, "_call_synthesizer_llm", fake_synth)
    monkeypatch.setattr(nodes, "_call_reflection_llm", fake_reflect)
