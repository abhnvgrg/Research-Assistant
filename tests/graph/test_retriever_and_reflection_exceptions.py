from __future__ import annotations

import app.graph.nodes as nodes
from app.graph.nodes import reflection_node, vector_retriever_node, web_search_node


async def test_vector_retriever_sets_degraded_on_pinecone_failure(decomposed_state, monkeypatch):
    async def failing_pinecone(sub_qs, user_id):
        raise ConnectionError("Pinecone unreachable")

    monkeypatch.setattr(nodes, "_call_vector_retriever", failing_pinecone)

    result = await vector_retriever_node(decomposed_state)

    assert result["degraded"] is True
    assert "vector_retrieval_failed" in result["error"]
    assert "chunks" not in result


async def test_web_search_node_catches_tavily_failure(decomposed_state, monkeypatch):
    async def failing_tavily(sub_qs):
        raise TimeoutError("Tavily request timed out")

    monkeypatch.setattr(nodes, "_call_web_search", failing_tavily)

    result = await web_search_node(decomposed_state)

    assert "web_search_failed" in result["error"]
    assert "chunks" not in result


async def test_reflection_node_fails_open_on_exception(decomposed_state, monkeypatch):
    async def failing_reflection_llm(query, sub_qs, answer):
        raise RuntimeError("OpenAI rate limited")

    monkeypatch.setattr(nodes, "_call_reflection_llm", failing_reflection_llm)
    decomposed_state["answer"] = "some answer"

    result = await reflection_node(decomposed_state)

    assert len(result["reflection_history"]) == 1
    entry = result["reflection_history"][0]
    assert entry["pass_"] is True
    assert "reflection_error" in entry["gap"]
