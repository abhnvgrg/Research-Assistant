"""
Shared fixtures.

Per the testing strategy we designed: unit tests never hit real
OpenAI/Pinecone/Tavily. Every test that needs a node's external call
monkeypatches the `_call_*` function in app.graph.nodes directly —
this is the same seam the REAL: comments point at, so monkeypatching
here is exactly where real API calls would later be swapped in.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

import app.graph.nodes as nodes_module
from app.graph.state import ResearchState
from app.quota_store import QuotaStore


@pytest.fixture(autouse=True)
def fresh_quota_store(monkeypatch):
    """quota_store is accessed via get_quota_store() (app/quota_store.py),
    not imported directly — nodes.py calls `from app.quota_store import
    get_quota_store` so it transparently picks up a real Supabase-backed
    store once app.db's pool is available. Tests patch the GETTER
    itself (same pattern as get_vector_store()/get_research_graph()
    elsewhere in this project), not a raw module attribute — without
    resetting it per test, usage recorded by one test would leak into
    every other test using the same user_id, causing order-dependent
    flakiness."""
    fresh = QuotaStore()
    monkeypatch.setattr(nodes_module, "get_quota_store", lambda: fresh)
    return fresh


@pytest.fixture
def base_state() -> ResearchState:
    """A minimal valid ResearchState — the required keys every node
    can assume exist, with the two accumulator lists initialized."""
    return {
        "query": "explain backpropagation",
        "user_id": "test-user-uuid",
        "run_id": str(uuid.uuid4()),
        "chunks": [],
        "reflection_history": [],
        "cycle_count": 0,
        "tokens_used": 0,
    }


@pytest.fixture
def decomposed_state(base_state: ResearchState) -> ResearchState:
    """A state that already has a valid decomposition — used by
    nodes downstream of decomposer_node (router, retrievers, grader)."""
    state = dict(base_state)
    state["decomposition"] = {
        "topic_identified": True,
        "multi_intent": False,
        "recency_required": False,
        "corrected_query": "Explain backpropagation",
        "sub_questions": [
            {"question": "What is backpropagation?", "intent": "general", "aliases": []},
            {"question": "How is the chain rule used in backpropagation?", "intent": "general", "aliases": []},
        ],
    }
    return state  # type: ignore[return-value]


def make_chunk(
    text: str = "sample chunk text",
    source: str = "https://example.com/doc",
    score: float = 0.8,
    origin: str = "web",
) -> dict[str, Any]:
    """Builds a single Chunk dict without going through retrieval."""
    return {
        "text": text,
        "source": source,
        "title": "Sample Title",
        "score": score,
        "origin": origin,
    }
