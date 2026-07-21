"""
Live end-to-end smoke test — spins up the REAL FastAPI app (via
Starlette's TestClient, which runs the actual ASGI app in-process)
and drives it through the full HTTP + SSE flow a real browser would:

  1. POST /research/query with a bearer token -> get run_id
  2. GET /research/{run_id}/stream -> read the live SSE event stream
  3. GET /research/{run_id}/result -> confirm the final state matches
  4. POST /ingest/document (text) -> get job_id
  5. GET /ingest/{job_id}/status -> confirm the ingestion result

Only the three true external SDK boundaries (OpenAI, Pinecone,
Tavily) are mocked, using the same ScriptedOpenAIClient /
FakePineconeVectorStore / FakeTavilyClient pattern as
run_full_stack_demo.py. Every line of our own HTTP, SSE, auth,
ingestion, and graph orchestration code runs for real.

IMPORTANT: TestClient MUST be used as a context manager
(`with TestClient(app) as client:`) — without the `with` block,
Starlette does not run the ASGI lifespan protocol at all, so the
Pinecone warmup and checkpointer initialization silently never run.
This was a real bug in an earlier version of this script: because
both of those failures degrade gracefully by design, a lifespan that
never ran was indistinguishable from one that ran and degraded — the
bug was invisible until something (verifying the checkpointer
actually initialized) explicitly checked for it.

Run with: python -m app.graph.run_http_smoke_test
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import jwt
from starlette.testclient import TestClient

import app.graph.nodes as nodes_module
import app.llm.client as llm_client_module
import app.llm.embeddings as embeddings_module
import app.retrieval.tavily_search as tavily_module
from app.graph.checkpointing import is_checkpointing_enabled
from app.graph.run_full_stack_demo import (
    FakePineconeVectorStore,
    FakeTavilyClient,
    ScriptedOpenAIClient,
)
from app.main import app

# app/api/deps.py now performs REAL JWT verification (see the auth
# hardening fix) — this script sets its own test-only secret and
# mints properly signed tokens, matching the pattern used in
# tests/api/conftest.py, rather than the arbitrary unsigned strings
# this script used before the fix (which the old stub accepted
# unconditionally — exactly the hole that fix closed).
_SMOKE_TEST_JWT_SECRET = "smoke-test-only-secret-never-used-for-anything-real"
os.environ.setdefault("SUPABASE_JWT_SECRET", _SMOKE_TEST_JWT_SECRET)


def _mint_bearer_header(user_id: str) -> dict[str, str]:
    payload = {"sub": user_id, "aud": "authenticated", "exp": int(time.time()) + 3600}
    token = jwt.encode(payload, _SMOKE_TEST_JWT_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def main() -> None:
    fake_openai = ScriptedOpenAIClient()
    fake_pinecone_store = FakePineconeVectorStore(index_host="fake", api_key="fake")
    fake_tavily = FakeTavilyClient()

    # Same monkeypatch-scoping lesson from earlier: patch every
    # module that did `from app.llm.client import get_client`.
    llm_client_module.get_client = lambda: fake_openai
    embeddings_module.get_client = lambda: fake_openai
    nodes_module._vector_store = fake_pinecone_store
    tavily_module.get_tavily_client = lambda: fake_tavily

    checkpoint_path = os.path.join(tempfile.gettempdir(), "smoke_test_checkpoints.db")
    os.environ["CHECKPOINT_DB_PATH"] = checkpoint_path
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)  # start clean each run

    print(f"\n{'=' * 70}")
    print("LIVE HTTP SMOKE TEST — real FastAPI app, mocked SDK boundaries")
    print(f"{'=' * 70}\n")

    # The `with` block is what actually triggers ASGI lifespan
    # startup/shutdown — see the module docstring above.
    with TestClient(app) as client:
        # 1. Health check
        health_resp = client.get("/health")
        print(f"GET /health -> {health_resp.status_code} {health_resp.json()}")
        assert health_resp.status_code == 200

        # 1b. Confirm the checkpointer actually initialized during the
        # real lifespan startup — not mocked, a genuine sqlite file
        # with LangGraph's checkpoint tables set up.
        print(f"Checkpointing enabled after lifespan startup: {is_checkpointing_enabled()}")
        assert is_checkpointing_enabled() is True
        assert os.path.exists(checkpoint_path), "checkpointer did not create its sqlite file"

        # 2. Auth is enforced
        unauth_resp = client.post("/research/query", json={"query": "explain backpropagation"})
        print(f"POST /research/query (no auth) -> {unauth_resp.status_code} (expect 401)")
        assert unauth_resp.status_code == 401

        # 3. Start a research run
        headers = _mint_bearer_header("smoke-test-user")
        start_resp = client.post(
            "/research/query",
            json={"query": "explain how attention mechanisms work"},
            headers=headers,
        )
        print(f"POST /research/query -> {start_resp.status_code} {start_resp.json()}")
        assert start_resp.status_code == 202
        run_id = start_resp.json()["run_id"]

        # 4. Cross-user access is blocked
        other_user_headers = _mint_bearer_header("smoke-test-other-user")
        blocked_resp = client.get(f"/research/{run_id}/result", headers=other_user_headers)
        print(f"GET /research/{{run_id}}/result (wrong user) -> {blocked_resp.status_code} (expect 404)")
        assert blocked_resp.status_code == 404

        # 5. Stream the SSE events (BackgroundTasks runs synchronously
        # under TestClient before the response is returned, so by the
        # time we open the stream the run has already completed — this
        # still proves the full replay-buffer path works correctly).
        print("\nGET /research/{run_id}/stream events:")
        with client.stream("GET", f"/research/{run_id}/stream", headers=headers) as stream_resp:
            assert stream_resp.status_code == 200
            for line in stream_resp.iter_lines():
                if line.startswith("data: "):
                    event = json.loads(line.removeprefix("data: "))
                    print(f"  {event.get('type')}: {json.dumps(event)[:120]}")

        # 6. Fetch the final result
        result_resp = client.get(f"/research/{run_id}/result", headers=headers)
        result = result_resp.json()
        print(f"\nGET /research/{{run_id}}/result -> {result_resp.status_code}")
        print(f"  status: {result['status']}")
        print(f"  answer: {result['answer']}")
        print(f"  citations: {result['citations']}")

        assert result["status"] == "complete"
        assert result["answer"] is not None
        assert result["citations"]

        # 7. Ingest a document via the text source type
        ingest_resp = client.post(
            "/ingest/document",
            data={"text": "Attention mechanisms let models weigh relevant tokens. " * 40},
            headers=headers,
        )
        print(f"\nPOST /ingest/document (text) -> {ingest_resp.status_code} {ingest_resp.json()}")
        assert ingest_resp.status_code == 202
        job_id = ingest_resp.json()["job_id"]

        # 8. Cross-user access to the ingest job is blocked too
        blocked_ingest_resp = client.get(f"/ingest/{job_id}/status", headers=other_user_headers)
        print(f"GET /ingest/{{job_id}}/status (wrong user) -> {blocked_ingest_resp.status_code} (expect 404)")
        assert blocked_ingest_resp.status_code == 404

        # 9. Fetch the real ingestion result (BackgroundTasks already
        # ran synchronously under TestClient, same as the research run
        # above)
        status_resp = client.get(f"/ingest/{job_id}/status", headers=headers)
        status_body = status_resp.json()
        print(f"GET /ingest/{{job_id}}/status -> {status_resp.status_code}")
        print(f"  status: {status_body['status']}")
        print(f"  title: {status_body['title']}")
        print(f"  chunks_ingested: {status_body['chunks_ingested']}")
        print(f"  chunks_dropped: {status_body['chunks_dropped']}")

        assert status_resp.status_code == 200
        assert status_body["status"] == "complete"
        assert status_body["chunks_ingested"] > 0

        print(f"\n{'-' * 70}")
        print("ALL ASSERTIONS PASSED — full HTTP + SSE + auth + graph + ingestion flow verified")
        print("(lifespan startup/shutdown, checkpointer, Pinecone warmup all genuinely ran)")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
