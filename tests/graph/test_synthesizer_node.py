"""
synthesizer_node tests.

The single most important test in this entire suite is
test_synthesizer_never_calls_llm_with_empty_graded — this is the
guard against edge case 3.3, the most dangerous silent failure we
identified: calling the synthesizer with zero retrieved evidence
produces a confident, uncited, fully hallucinated answer that is
visually indistinguishable from a real one.
"""

from __future__ import annotations

import app.graph.nodes as nodes
from app.graph.nodes import synthesizer_node
from tests.conftest import make_chunk


async def test_synthesizer_never_calls_llm_with_empty_graded(decomposed_state, monkeypatch):
    """If state['graded'] is empty, synthesizer_node must return a
    graceful 'insufficient sources' message WITHOUT ever invoking
    the LLM. We assert this by making the mocked LLM call itself
    raise — if it's ever called, the test fails immediately."""

    decomposed_state["graded"] = []

    async def llm_must_not_be_called(query, graded):
        raise AssertionError(
            "synthesizer called the LLM with empty graded chunks — "
            "this is the hallucination-on-empty-context bug (edge case 3.3)"
        )

    monkeypatch.setattr(nodes, "_call_synthesizer_llm", llm_must_not_be_called)

    result = await synthesizer_node(decomposed_state)

    assert result["citations"] == {}
    assert "insufficient sources" in result["answer"].lower() or "wasn't able" in result["answer"].lower()
    assert result["tokens_used"] == 0  # no LLM call made — nothing to charge for


async def test_synthesizer_with_graded_chunks_calls_llm_and_returns_citations(
    decomposed_state, monkeypatch
):
    graded = [
        {**make_chunk("fact A", source="https://a.com"), "relevant": True, "grade_score": 0.9, "grade_reason": "ok"},
        {**make_chunk("fact B", source="https://b.com"), "relevant": True, "grade_score": 0.8, "grade_reason": "ok"},
    ]
    decomposed_state["graded"] = graded

    async def fake_synth(query, graded_chunks):
        assert len(graded_chunks) == 2
        return {
            "answer": "Synthesized answer [1] [2]",
            "citations": {"1": "https://a.com", "2": "https://b.com"},
            "tokens_used": 890,
        }

    monkeypatch.setattr(nodes, "_call_synthesizer_llm", fake_synth)

    result = await synthesizer_node(decomposed_state)

    assert result["citations"] == {"1": "https://a.com", "2": "https://b.com"}
    assert result["error"] is None
    assert result["tokens_used"] == 890


async def test_synthesizer_catches_llm_exception(decomposed_state, monkeypatch):
    decomposed_state["graded"] = [
        {**make_chunk(), "relevant": True, "grade_score": 0.9, "grade_reason": "ok"}
    ]

    async def failing_synth(query, graded_chunks):
        raise RuntimeError("OpenAI 503")

    monkeypatch.setattr(nodes, "_call_synthesizer_llm", failing_synth)

    result = await synthesizer_node(decomposed_state)

    assert "synthesis_failed" in result["error"]
    assert "answer" not in result  # node must not fabricate a partial answer on failure
    assert result["tokens_used"] == 0
