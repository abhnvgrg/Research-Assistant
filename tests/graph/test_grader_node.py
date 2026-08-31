from __future__ import annotations

import asyncio

import app.graph.nodes as nodes
from app.graph.nodes import GRADE_THRESHOLD, MAX_CHUNKS_TO_SYNTH, grader_node
from tests.conftest import make_chunk


async def test_grader_drops_chunks_below_threshold(decomposed_state, monkeypatch):
    decomposed_state["chunks"] = [make_chunk("relevant"), make_chunk("irrelevant")]

    scores = iter([0.9, 0.2])

    async def fake_grade(chunk, query):
        return {"relevant": True, "score": next(scores), "reason": "stub"}, 15

    monkeypatch.setattr(nodes, "_call_grader_llm", fake_grade)

    result = await grader_node(decomposed_state)

    assert len(result["graded"]) == 1
    assert result["graded"][0]["text"] == "relevant"
    assert all(g["grade_score"] >= GRADE_THRESHOLD for g in result["graded"])


async def test_grader_sorts_by_score_descending(decomposed_state, monkeypatch):
    decomposed_state["chunks"] = [
        make_chunk("low", score=0.6),
        make_chunk("high", score=0.6),
        make_chunk("mid", score=0.6),
    ]

    score_map = {"low": 0.55, "high": 0.95, "mid": 0.70}

    async def fake_grade(chunk, query):
        return {"relevant": True, "score": score_map[chunk["text"]], "reason": "stub"}, 15

    monkeypatch.setattr(nodes, "_call_grader_llm", fake_grade)

    result = await grader_node(decomposed_state)
    texts_in_order = [g["text"] for g in result["graded"]]

    assert texts_in_order == ["high", "mid", "low"]


async def test_grader_caps_at_max_chunks_to_synth(decomposed_state, monkeypatch):
    decomposed_state["chunks"] = [make_chunk(f"chunk-{i}") for i in range(20)]

    async def fake_grade(chunk, query):
        return {"relevant": True, "score": 0.9, "reason": "stub"}, 15

    monkeypatch.setattr(nodes, "_call_grader_llm", fake_grade)

    result = await grader_node(decomposed_state)

    assert len(result["graded"]) == MAX_CHUNKS_TO_SYNTH


async def test_grader_calls_run_concurrently(decomposed_state, monkeypatch):
    decomposed_state["chunks"] = [make_chunk(f"chunk-{i}") for i in range(10)]

    async def slow_grade(chunk, query):
        await asyncio.sleep(0.05)
        return {"relevant": True, "score": 0.9, "reason": "stub"}, 15

    monkeypatch.setattr(nodes, "_call_grader_llm", slow_grade)

    loop = asyncio.get_event_loop()
    start = loop.time()
    await grader_node(decomposed_state)
    elapsed = loop.time() - start

    assert elapsed < 0.25, f"grading took {elapsed:.3f}s — looks sequential, not parallel"


async def test_grader_sums_tokens_used_across_all_chunks(decomposed_state, monkeypatch):
    decomposed_state["chunks"] = [make_chunk(f"chunk-{i}") for i in range(4)]

    async def fake_grade(chunk, query):
        return {"relevant": True, "score": 0.9, "reason": "stub"}, 25

    monkeypatch.setattr(nodes, "_call_grader_llm", fake_grade)

    result = await grader_node(decomposed_state)

    assert result["tokens_used"] == 100


async def test_grader_degrades_to_empty_list_on_exception(decomposed_state, monkeypatch):
    decomposed_state["chunks"] = [make_chunk()]

    async def failing_grade(chunk, query):
        raise ValueError("malformed grader JSON")

    monkeypatch.setattr(nodes, "_call_grader_llm", failing_grade)

    result = await grader_node(decomposed_state)

    assert result["graded"] == []
    assert result["tokens_used"] == 0
    assert "grading_failed" in result["error"]


async def test_grader_handles_empty_chunk_list(decomposed_state, monkeypatch):
    decomposed_state["chunks"] = []

    async def fake_grade(chunk, query):
        return {"relevant": True, "score": 0.9, "reason": "stub"}, 15

    monkeypatch.setattr(nodes, "_call_grader_llm", fake_grade)

    result = await grader_node(decomposed_state)

    assert result["graded"] == []
