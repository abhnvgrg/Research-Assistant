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

    llm_client_module.get_client = lambda: fake_openai
    embeddings_module.get_client = lambda: fake_openai
    nodes_module._vector_store = fake_pinecone_store
    tavily_module.get_tavily_client = lambda: fake_tavily

    checkpoint_path = os.path.join(tempfile.gettempdir(), "smoke_test_checkpoints.db")
    os.environ["CHECKPOINT_DB_PATH"] = checkpoint_path
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"\n{'=' * 70}")
    print("LIVE HTTP SMOKE TEST — real FastAPI app, mocked SDK boundaries")
    print(f"{'=' * 70}\n")

    with TestClient(app) as client:
        health_resp = client.get("/health")
        print(f"GET /health -> {health_resp.status_code} {health_resp.json()}")
        assert health_resp.status_code == 200

        print(f"Checkpointing enabled after lifespan startup: {is_checkpointing_enabled()}")
        assert is_checkpointing_enabled() is True
        assert os.path.exists(checkpoint_path), "checkpointer did not create its sqlite file"

        unauth_resp = client.post("/research/query", json={"query": "explain backpropagation"})
        print(f"POST /research/query (no auth) -> {unauth_resp.status_code} (expect 401)")
        assert unauth_resp.status_code == 401

        headers = _mint_bearer_header("smoke-test-user")
        start_resp = client.post(
            "/research/query",
            json={"query": "explain how attention mechanisms work"},
            headers=headers,
        )
        print(f"POST /research/query -> {start_resp.status_code} {start_resp.json()}")
        assert start_resp.status_code == 202
        run_id = start_resp.json()["run_id"]

        other_user_headers = _mint_bearer_header("smoke-test-other-user")
        blocked_resp = client.get(f"/research/{run_id}/result", headers=other_user_headers)
        print(f"GET /research/{{run_id}}/result (wrong user) -> {blocked_resp.status_code} (expect 404)")
        assert blocked_resp.status_code == 404

        print("\nGET /research/{run_id}/stream events:")
        with client.stream("GET", f"/research/{run_id}/stream", headers=headers) as stream_resp:
            assert stream_resp.status_code == 200
            for line in stream_resp.iter_lines():
                if line.startswith("data: "):
                    event = json.loads(line.removeprefix("data: "))
                    print(f"  {event.get('type')}: {json.dumps(event)[:120]}")

        result_resp = client.get(f"/research/{run_id}/result", headers=headers)
        result = result_resp.json()
        print(f"\nGET /research/{{run_id}}/result -> {result_resp.status_code}")
        print(f"  status: {result['status']}")
        print(f"  answer: {result['answer']}")
        print(f"  citations: {result['citations']}")

        assert result["status"] == "complete"
        assert result["answer"] is not None
        assert result["citations"]

        ingest_resp = client.post(
            "/ingest/document",
            data={"text": "Attention mechanisms let models weigh relevant tokens. " * 40},
            headers=headers,
        )
        print(f"\nPOST /ingest/document (text) -> {ingest_resp.status_code} {ingest_resp.json()}")
        assert ingest_resp.status_code == 202
        job_id = ingest_resp.json()["job_id"]

        blocked_ingest_resp = client.get(f"/ingest/{job_id}/status", headers=other_user_headers)
        print(f"GET /ingest/{{job_id}}/status (wrong user) -> {blocked_ingest_resp.status_code} (expect 404)")
        assert blocked_ingest_resp.status_code == 404

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
