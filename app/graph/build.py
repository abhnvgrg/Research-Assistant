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


def check_quota(state: ResearchState) -> Literal["ok", "fail"]:
    return "ok" if state.get("quota_ok", False) else "fail"


def check_decomposition(state: ResearchState) -> Literal["ok", "fail"]:
    decomp = state.get("decomposition")
    if state.get("error"):
        return "fail"
    if not decomp or not decomp.get("topic_identified", False):
        return "fail"
    if decomp.get("multi_intent", False):
        return "fail"
    return "ok"


def pick_sources(state: ResearchState) -> str | list[str]:
    route = state.get("route", "both")
    if route == "vector":
        return "vector_retriever"
    if route == "web":
        return "web_search"
    return ["vector_retriever", "web_search"]


def route_after_reflection(state: ResearchState) -> Literal["router", "formatter"]:
    if state.get("error"):
        return "formatter"

    cycle_count = state.get("cycle_count", 0)
    history = state.get("reflection_history", [])
    last_reflection = history[-1] if history else {"pass_": True}

    if last_reflection.get("pass_", True):
        return "formatter"
    if cycle_count >= MAX_CYCLES:
        return "formatter"
    return "router"


def build_research_graph(checkpointer=None, interrupt_after=None):
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

    builder.add_conditional_edges("router", pick_sources)

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


research_graph = build_research_graph()
