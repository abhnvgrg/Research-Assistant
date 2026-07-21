"""
grader_node tests.

This is the most heavily-scrutinized node in our design — covers:
  - chunks below GRADE_THRESHOLD are dropped
  - passed chunks are sorted by grade_score descending
  - result is capped at MAX_CHUNKS_TO_SYNTH (edge case: too many
    relevant chunks would blow the synthesizer's context budget)
  - grading calls run concurrently, not sequentially (asyncio.gather)
  - an exception during grading degrades to graded=[] rather than
    crashing — this is what lets synthesizer_node's empty-graded
    guard (edge case 3.3) kick in correctly
"""

from __future__ import annotations

import asyncio

import app.graph.nodes as nodes
from app.graph.nodes import GRADE_THRESHOLD, MAX_CHUNKS_TO_SYNTH, grader_node
from tests.conftest import make_chunk


async def test_grader_drops_chunks_below_threshold(decomposed_state, monkeypatch):
    decomposed_state["chunks"] = [make_chunk("relevant"), make_chunk("irrelevant")]

    scores = iter([0.9, 0.2])  # second chunk fails GRADE_THRESHOLD (0.5)

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
    """Edge case: more than MAX_CHUNKS_TO_SYNTH chunks pass grading —
    only the top N by score should survive, protecting the
    synthesizer's context window budget (silent failure S5 in our
    edge case catalogue)."""
    decomposed_state["chunks"] = [make_chunk(f"chunk-{i}") for i in range(20)]

    async def fake_grade(chunk, query):
        return {"relevant": True, "score": 0.9, "reason": "stub"}, 15

    monkeypatch.setattr(nodes, "_call_grader_llm", fake_grade)

    result = await grader_node(decomposed_state)

    assert len(result["graded"]) == MAX_CHUNKS_TO_SYNTH


async def test_grader_calls_run_concurrently(decomposed_state, monkeypatch):
    """Proves grading is NOT sequential — N chunks graded with a
    50ms simulated delay each should take ~50ms total, not N*50ms.
    This is the asyncio.gather() pattern from our design; sequential
    grading would multiply latency directly by chunk count."""
    decomposed_state["chunks"] = [make_chunk(f"chunk-{i}") for i in range(10)]

    async def slow_grade(chunk, query):
        await asyncio.sleep(0.05)
        return {"relevant": True, "score": 0.9, "reason": "stub"}, 15

    monkeypatch.setattr(nodes, "_call_grader_llm", slow_grade)

    loop = asyncio.get_event_loop()
    start = loop.time()
    await grader_node(decomposed_state)
    elapsed = loop.time() - start

    # Sequential would take >= 0.5s (10 * 0.05s). Parallel should be
    # well under that — generous threshold to avoid CI flakiness.
    assert elapsed < 0.25, f"grading took {elapsed:.3f}s — looks sequential, not parallel"


async def test_grader_sums_tokens_used_across_all_chunks(decomposed_state, monkeypatch):
    """grader_node must SUM the token cost of every chunk graded in
    this pass, not just report the last one's — this is what feeds
    the tokens_used operator.add reducer correctly across a full run."""
    decomposed_state["chunks"] = [make_chunk(f"chunk-{i}") for i in range(4)]

    async def fake_grade(chunk, query):
        return {"relevant": True, "score": 0.9, "reason": "stub"}, 25

    monkeypatch.setattr(nodes, "_call_grader_llm", fake_grade)

    result = await grader_node(decomposed_state)

    assert result["tokens_used"] == 100  # 4 chunks * 25 tokens each


async def test_grader_degrades_to_empty_list_on_exception(decomposed_state, monkeypatch):
    """If grading itself throws (e.g. malformed JSON from the LLM
    that survives retries), the node must not propagate the
    exception — it must return graded=[] so synthesizer_node's
    empty-context guard (the most dangerous silent failure in the
    whole system) can catch it downstream."""
    decomposed_state["chunks"] = [make_chunk()]

    async def failing_grade(chunk, query):
        raise ValueError("malformed grader JSON")

    monkeypatch.setattr(nodes, "_call_grader_llm", failing_grade)

    result = await grader_node(decomposed_state)

    assert result["graded"] == []
    assert result["tokens_used"] == 0
    assert "grading_failed" in result["error"]


async def test_grader_handles_empty_chunk_list(decomposed_state, monkeypatch):
    """No chunks retrieved at all (e.g. Pinecone down AND Tavily
    returned nothing) — grader must not crash on an empty gather()."""
    decomposed_state["chunks"] = []

    async def fake_grade(chunk, query):
        return {"relevant": True, "score": 0.9, "reason": "stub"}, 15

    monkeypatch.setattr(nodes, "_call_grader_llm", fake_grade)

    result = await grader_node(decomposed_state)

    assert result["graded"] == []
