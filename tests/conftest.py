from __future__ import annotations

import uuid
from typing import Any

import pytest

import app.graph.nodes as nodes_module
from app.graph.state import ResearchState
from app.quota_store import QuotaStore


@pytest.fixture(autouse=True)
def fresh_quota_store(monkeypatch):
    fresh = QuotaStore()
    monkeypatch.setattr(nodes_module, "get_quota_store", lambda: fresh)
    return fresh


@pytest.fixture
def base_state() -> ResearchState:
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
    return {
        "text": text,
        "source": source,
        "title": "Sample Title",
        "score": score,
        "origin": origin,
    }
