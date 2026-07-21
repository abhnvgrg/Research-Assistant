"""
router_node tests.

Covers: route selection based on recency_required, and that
cycle_count is correctly incremented on every pass through the
router (this is what feeds route_after_reflection's MAX_CYCLES cap).
"""

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
    """Simulates the router being re-entered after a failed
    reflection — cycle_count must keep climbing, not reset."""
    decomposed_state["cycle_count"] = 2
    result = await router_node(decomposed_state)
    assert result["cycle_count"] == 3
