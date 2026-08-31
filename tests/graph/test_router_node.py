from __future__ import annotations

from app.graph.nodes import router_node


async def test_router_routes_both_by_default(decomposed_state):
    result = await router_node(decomposed_state)
    assert result["route"] == "both"


async def test_router_routes_web_only_when_recency_required(decomposed_state):
    decomposed_state["decomposition"]["recency_required"] = True
    result = await router_node(decomposed_state)
    assert result["route"] == "web"


async def test_router_increments_cycle_count_from_zero(decomposed_state):
    decomposed_state["cycle_count"] = 0
    result = await router_node(decomposed_state)
    assert result["cycle_count"] == 1


async def test_router_increments_cycle_count_across_loops(decomposed_state):
    decomposed_state["cycle_count"] = 2
    result = await router_node(decomposed_state)
    assert result["cycle_count"] == 3
