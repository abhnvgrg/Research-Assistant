from __future__ import annotations

import pytest

import app.graph.nodes as nodes
from app.graph.nodes import decomposer_node


async def test_decomposer_returns_expected_keys(base_state, monkeypatch):
    async def fake_llm(query: str) -> tuple[dict, int]:
        return {
            "topic_identified": True,
            "multi_intent": False,
            "recency_required": False,
            "corrected_query": "Explain backpropagation",
            "sub_questions": [
                {"question": "What is backpropagation?", "intent": "general", "aliases": []}
            ],
        }, 123

    monkeypatch.setattr(nodes, "_call_decomposer_llm", fake_llm)

    result = await decomposer_node(base_state)

    assert "decomposition" in result
    assert result["decomposition"]["topic_identified"] is True
    assert result["error"] is None
    assert result["tokens_used"] == 123


async def test_decomposer_node_returns_partial_dict_not_full_state(base_state, monkeypatch):
    async def fake_llm(query: str) -> tuple[dict, int]:
        return {
            "topic_identified": True,
            "multi_intent": False,
            "recency_required": False,
            "corrected_query": query,
            "sub_questions": [],
        }, 50

    monkeypatch.setattr(nodes, "_call_decomposer_llm", fake_llm)

    result = await decomposer_node(base_state)

    assert set(result.keys()) == {"decomposition", "tokens_used", "error"}


async def test_decomposer_node_catches_llm_exception(base_state, monkeypatch):
    async def failing_llm(query: str) -> tuple[dict, int]:
        raise TimeoutError("OpenAI request timed out")

    monkeypatch.setattr(nodes, "_call_decomposer_llm", failing_llm)

    result = await decomposer_node(base_state)

    assert "decomposition" not in result
    assert "decomposition_failed" in result["error"]
    assert "OpenAI request timed out" in result["error"]
    assert result["tokens_used"] == 0
