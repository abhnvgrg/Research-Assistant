"""
Graph construction — wires every node from nodes.py into the full
agent loop with conditional edges.

This is the literal implementation of the architecture diagram:

    quota_check -> decomposer -> router -> (vector | web | both) -> grader
        -> synthesizer -> reflection -> [loop back to router OR formatter]

Two things worth re-reading before touching this file:

1. The "both" branch in pick_sources returns a LIST of node names.
   That's a fan-out — LangGraph runs vector_retriever_node and
   web_search_node in the SAME superstep, in parallel. They must
   never depend on each other's output.

2. Both vector_retriever and web_search have an add_edge straight to
   "grader". Two edges into the same node is an implicit join —
   LangGraph waits for both branches before running grader once.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from app.graph.nodes import (
    MAX_CYCLES,
    decomposer_node,
    error_handler_node,
    formatter_node,
    grader_node,
    quota_check_node,
    reflection_node,
    router_node,
    synthesizer_node,
    vector_retriever_node,
    web_search_node,
)
from app.graph.state import ResearchState


# --------------------------------------------------------------------------
# Conditional edge functions — pure functions of state, return a
# string that must exactly match a node name (or a key in the
# path_map passed to add_conditional_edges).
# --------------------------------------------------------------------------

def check_quota(state: ResearchState) -> Literal["ok", "fail"]:
    return "ok" if state.get("quota_ok", False) else "fail"


def check_decomposition(state: ResearchState) -> Literal["ok", "fail"]:
    """Reject before any retrieval if the decomposer couldn't find
    a topic, or if the query had multiple unrelated topics fused
    together (edge cases 1.1 and 1.3)."""
    decomp = state.get("decomposition")
    if state.get("error"):
        return "fail"
    if not decomp or not decomp.get("topic_identified", False):
        return "fail"
    if decomp.get("multi_intent", False):
        return "fail"
    return "ok"


def pick_sources(state: ResearchState) -> str | list[str]:
    """Returns node name(s) to run next. A list triggers a LangGraph
    fan-out — vector_retriever and web_search run in the same
    superstep when route == 'both'."""
    route = state.get("route", "both")
    if route == "vector":
        return "vector_retriever"
    if route == "web":
        return "web_search"
    return ["vector_retriever", "web_search"]


def route_after_reflection(state: ResearchState) -> Literal["router", "formatter"]:
    """The loop-or-exit decision. Hard cap at MAX_CYCLES regardless
    of what reflection says — this is what prevents infinite
    reflection thrashing (edge case 5.1)."""
    if state.get("error"):
        return "formatter"  # still format — formatter handles error display

    cycle_count = state.get("cycle_count", 0)
    history = state.get("reflection_history", [])
    last_reflection = history[-1] if history else {"pass_": True}

    if last_reflection.get("pass_", True):
        return "formatter"
    if cycle_count >= MAX_CYCLES:
        return "formatter"
    return "router"


# --------------------------------------------------------------------------
# Build the graph
# --------------------------------------------------------------------------

def build_research_graph(checkpointer=None, interrupt_after=None):
    """checkpointer=None (the default) compiles a graph with no
    persistence — every existing test and demo script builds/imports
    the module-level `research_graph` below, which stays exactly as
    it was. Passing a real checkpointer (see app/graph/checkpointing.py)
    is additive: it enables state to be persisted after every
    superstep, keyed by thread_id, so a run can be resumed from its
    last checkpoint rather than restarting from scratch — used by the
    real FastAPI app via its lifespan, not by tests.

    interrupt_after is a list of node names LangGraph will pause
    after (requires a checkpointer to be meaningful) — used only by
    tests/test_checkpointing.py to simulate a process crash mid-run
    without needing to actually kill and restart anything."""
    builder = StateGraph(ResearchState)

    builder.add_node("quota_check", quota_check_node)
    builder.add_node("decomposer", decomposer_node)
    builder.add_node("router", router_node)
    builder.add_node("vector_retriever", vector_retriever_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("grader", grader_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("reflection", reflection_node)
    builder.add_node("formatter", formatter_node)
    builder.add_node("error_handler", error_handler_node)

    builder.set_entry_point("quota_check")

    builder.add_conditional_edges(
        "quota_check",
        check_quota,
        {"ok": "decomposer", "fail": "error_handler"},
    )

    builder.add_conditional_edges(
        "decomposer",
        check_decomposition,
        {"ok": "router", "fail": "error_handler"},
    )

    # No path_map here: pick_sources already returns real node names
    # (or a list of them for fan-out), so LangGraph routes directly.
    builder.add_conditional_edges("router", pick_sources)

    # Implicit join: both branches converge here. LangGraph waits for
    # whichever subset of {vector_retriever, web_search} was launched.
    builder.add_edge("vector_retriever", "grader")
    builder.add_edge("web_search", "grader")

    builder.add_edge("grader", "synthesizer")
    builder.add_edge("synthesizer", "reflection")

    builder.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {"router": "router", "formatter": "formatter"},
    )

    builder.add_edge("formatter", END)
    builder.add_edge("error_handler", END)

    return builder.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)


# Module-level compiled graph, no checkpointer — imported by tests and
# demo scripts, and used as the FastAPI app's fallback if the real
# checkpointer hasn't been initialized yet (or failed to initialize).
research_graph = build_research_graph()
