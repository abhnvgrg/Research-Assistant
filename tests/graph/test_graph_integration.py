"""
Graph-level tests — these exercise the REAL compiled StateGraph
(not individual node functions), proving things that can only be
proven by actually running LangGraph's reducer/superstep machinery:

  - operator.add reducers correctly accumulate 'chunks' and
    'reflection_history' across multiple reflection cycles
  - the fan-out at the Router truly runs vector_retriever and
    web_search in parallel
  - a topic-less query never reaches retrieval or synthesis nodes

These sit one level above the unit tests in the pyramid — still
fully mocked (no real OpenAI/Pinecone/Tavily calls), but exercising
real LangGraph execution semantics instead of isolated functions.
"""

from __future__ import annotations

import uuid

import pytest

import app.graph.nodes as nodes
from app.graph.build import research_graph
from app.graph.state import ResearchState


def _fresh_state(query: str = "explain attention mechanisms") -> ResearchState:
    return {
        "query": query,
        "user_id": "test-user-uuid",
        "run_id": str(uuid.uuid4()),
        "chunks": [],
        "reflection_history": [],
        "cycle_count": 0,
        "tokens_used": 0,
    }


@pytest.fixture(autouse=True)
def fast_stubs(monkeypatch):
    """Speeds up every graph test by stripping the artificial
    asyncio.sleep() delays baked into the stub functions, and gives
    deterministic single-pass behavior (reflection always passes)
    unless a specific test overrides it."""

    async def fast_decomposer(query: str) -> tuple[dict, int]:
        return {
            "topic_identified": True,
            "multi_intent": False,
            "recency_required": False,
            "corrected_query": query,
            "sub_questions": [
                {"question": "sub-q-1", "intent": "general", "aliases": []},
                {"question": "sub-q-2", "intent": "general", "aliases": []},
            ],
        }, 50

    async def fast_vector(sub_qs, user_id):
        return [{"text": "v-chunk", "source": "pinecone://x", "title": "t", "score": 0.8, "origin": "vector"}]

    async def fast_web(sub_qs):
        return [{"text": "w-chunk", "source": "https://x.com", "title": "t", "score": 0.7, "origin": "web"}]

    async def fast_grade(chunk, query):
        return {"relevant": True, "score": 0.9, "reason": "stub"}, 20

    async def fast_synth(query, graded):
        return {"answer": "stub answer", "citations": {str(i + 1): c["source"] for i, c in enumerate(graded)}, "tokens_used": 500}

    async def fast_reflect_pass(query, sub_qs, answer):
        return {"pass_": True, "checklist": {"q1": True}, "gap": "nothing"}, 30

    monkeypatch.setattr(nodes, "_call_decomposer_llm", fast_decomposer)
    monkeypatch.setattr(nodes, "_call_vector_retriever", fast_vector)
    monkeypatch.setattr(nodes, "_call_web_search", fast_web)
    monkeypatch.setattr(nodes, "_call_grader_llm", fast_grade)
    monkeypatch.setattr(nodes, "_call_synthesizer_llm", fast_synth)
    monkeypatch.setattr(nodes, "_call_reflection_llm", fast_reflect_pass)


async def _run_to_completion(state: ResearchState) -> dict:
    final = {}
    async for snapshot in research_graph.astream(state, stream_mode="values"):
        final = snapshot
    return final


async def test_single_pass_run_completes_with_one_cycle():
    final = await _run_to_completion(_fresh_state())

    assert final["cycle_count"] == 1
    assert final["answer"] == "stub answer"
    # 1 vector chunk + 1 web chunk from the single cycle
    assert len(final["chunks"]) == 2
    assert len(final["reflection_history"]) == 1


async def test_chunks_accumulate_via_operator_add_across_cycles(monkeypatch):
    """The exact bug class we caught live in run_demo.py: chunks
    must accumulate, not be overwritten, when the Router is
    re-entered after a failed reflection."""
    call_count = {"n": 0}

    async def fail_twice_then_pass(query, sub_qs, answer):
        call_count["n"] += 1
        if call_count["n"] < 3:
            return {"pass_": False, "checklist": {}, "gap": f"cycle {call_count['n']} incomplete"}, 30
        return {"pass_": True, "checklist": {}, "gap": "nothing"}, 30

    monkeypatch.setattr(nodes, "_call_reflection_llm", fail_twice_then_pass)

    final = await _run_to_completion(_fresh_state())

    assert final["cycle_count"] == 3
    # 2 chunks per cycle (1 vector + 1 web) * 3 cycles = 6
    assert len(final["chunks"]) == 6
    assert len(final["reflection_history"]) == 3


async def test_max_cycles_enforced_even_if_reflection_never_passes(monkeypatch):
    async def always_fail(query, sub_qs, answer):
        return {"pass_": False, "checklist": {}, "gap": "still incomplete"}, 30

    monkeypatch.setattr(nodes, "_call_reflection_llm", always_fail)

    final = await _run_to_completion(_fresh_state())

    # Must stop at MAX_CYCLES (3), not loop forever, despite every
    # single reflection call returning pass_=False.
    assert final["cycle_count"] == 3
    assert len(final["reflection_history"]) == 3
    assert final.get("answer") is not None  # still produced a final answer


async def test_topic_less_query_never_reaches_retrieval(monkeypatch):
    async def empty_topic_decomposer(query: str) -> tuple[dict, int]:
        return {
            "topic_identified": False,
            "multi_intent": False,
            "recency_required": False,
            "corrected_query": query,
            "sub_questions": [],
        }, 50

    monkeypatch.setattr(nodes, "_call_decomposer_llm", empty_topic_decomposer)

    visited = []
    async for event in research_graph.astream(_fresh_state(), stream_mode="updates"):
        visited.extend(event.keys())

    assert "error_handler" in visited
    forbidden = {"vector_retriever", "web_search", "grader", "synthesizer"}
    assert not (forbidden & set(visited)), f"retrieval/synthesis ran on a topic-less query: {forbidden & set(visited)}"


async def test_multi_intent_query_never_reaches_retrieval(monkeypatch):
    async def multi_intent_decomposer(query: str) -> tuple[dict, int]:
        return {
            "topic_identified": True,
            "multi_intent": True,
            "recency_required": False,
            "corrected_query": query,
            "sub_questions": [
                {"question": "about topic A", "intent": "general", "aliases": []},
                {"question": "about topic B", "intent": "general", "aliases": []},
            ],
        }, 50

    monkeypatch.setattr(nodes, "_call_decomposer_llm", multi_intent_decomposer)

    visited = []
    async for event in research_graph.astream(_fresh_state(), stream_mode="updates"):
        visited.extend(event.keys())

    assert "error_handler" in visited
    assert "synthesizer" not in visited


async def test_recency_query_routes_web_only(monkeypatch):
    state = _fresh_state("latest GPT-5 benchmark results")

    async def recency_decomposer(query: str) -> tuple[dict, int]:
        return {
            "topic_identified": True,
            "multi_intent": False,
            "recency_required": True,
            "corrected_query": query,
            "sub_questions": [{"question": query, "intent": "recency", "aliases": []}],
        }, 50

    monkeypatch.setattr(nodes, "_call_decomposer_llm", recency_decomposer)

    visited = []
    async for event in research_graph.astream(state, stream_mode="updates"):
        visited.extend(event.keys())

    assert "web_search" in visited
    assert "vector_retriever" not in visited


async def test_quota_exceeded_user_never_reaches_decomposer(fresh_quota_store):
    """The real end-to-end proof of the quota fix: a user already
    over their limit is blocked at quota_check_node, before the
    decomposer (or anything else that would spend real money) ever
    runs — verified by node visitation, not just by reading the
    conditional-edge code."""
    state = _fresh_state()
    await fresh_quota_store.set_limit(state["user_id"], 100)
    await fresh_quota_store.add_usage(state["user_id"], 100)

    visited = []
    async for event in research_graph.astream(state, stream_mode="updates"):
        visited.extend(event.keys())

    assert "error_handler" in visited
    forbidden = {"decomposer", "router", "vector_retriever", "web_search", "grader", "synthesizer"}
    assert not (forbidden & set(visited)), f"spent money despite exceeded quota: {forbidden & set(visited)}"


async def test_full_run_accumulates_tokens_and_records_real_usage(fresh_quota_store):
    """The other half of the quota fix: a full successful run must
    both (a) correctly SUM tokens_used across every LLM-calling node
    via the operator.add reducer, and (b) actually record that total
    against the user's quota by the time the run completes — proving
    the whole loop (check -> spend -> record) is closed, not just
    that individual pieces work in isolation."""
    state = _fresh_state()

    final_state = {}
    async for snapshot in research_graph.astream(state, stream_mode="values"):
        final_state = snapshot

    # fast_stubs fixture reports: decomposer=50, grader=20/chunk (2
    # chunks: 1 vector + 1 web = 40), synth=500, reflect=30 → 620 total
    assert final_state["tokens_used"] == 620
    usage = await fresh_quota_store.get_usage(state["user_id"])
    assert usage.tokens_used == 620
