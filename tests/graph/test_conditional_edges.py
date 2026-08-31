from __future__ import annotations

from app.graph.build import (
    MAX_CYCLES,
    check_decomposition,
    check_quota,
    pick_sources,
    route_after_reflection,
)


def test_check_quota_ok(base_state):
    base_state["quota_ok"] = True
    assert check_quota(base_state) == "ok"


def test_check_quota_fail_when_false(base_state):
    base_state["quota_ok"] = False
    assert check_quota(base_state) == "fail"


def test_check_quota_fail_when_missing(base_state):
    assert "quota_ok" not in base_state
    assert check_quota(base_state) == "fail"


def test_check_decomposition_ok(decomposed_state):
    assert check_decomposition(decomposed_state) == "ok"


def test_check_decomposition_fails_on_no_topic(decomposed_state):
    decomposed_state["decomposition"]["topic_identified"] = False
    assert check_decomposition(decomposed_state) == "fail"


def test_check_decomposition_fails_on_multi_intent(decomposed_state):
    decomposed_state["decomposition"]["multi_intent"] = True
    assert check_decomposition(decomposed_state) == "fail"


def test_check_decomposition_fails_if_decomposer_errored(base_state):
    base_state["error"] = "decomposition_failed: timeout"
    assert check_decomposition(base_state) == "fail"


def test_pick_sources_vector_only(base_state):
    base_state["route"] = "vector"
    assert pick_sources(base_state) == "vector_retriever"


def test_pick_sources_web_only(base_state):
    base_state["route"] = "web"
    assert pick_sources(base_state) == "web_search"


def test_pick_sources_both_returns_list_for_fanout(base_state):
    base_state["route"] = "both"
    result = pick_sources(base_state)
    assert isinstance(result, list)
    assert set(result) == {"vector_retriever", "web_search"}


def test_route_after_reflection_exits_on_pass(base_state):
    base_state["cycle_count"] = 1
    base_state["reflection_history"] = [{"pass_": True, "checklist": {}, "gap": "nothing"}]
    assert route_after_reflection(base_state) == "formatter"


def test_route_after_reflection_loops_on_fail_under_max_cycles(base_state):
    base_state["cycle_count"] = 1
    base_state["reflection_history"] = [{"pass_": False, "checklist": {}, "gap": "missing X"}]
    assert route_after_reflection(base_state) == "router"


def test_route_after_reflection_hard_caps_at_max_cycles(base_state):
    base_state["cycle_count"] = MAX_CYCLES
    base_state["reflection_history"] = [{"pass_": False, "checklist": {}, "gap": "still missing X"}]
    assert route_after_reflection(base_state) == "formatter"


def test_route_after_reflection_caps_even_with_many_fails_beyond_max(base_state):
    base_state["cycle_count"] = MAX_CYCLES + 5
    base_state["reflection_history"] = [{"pass_": False, "checklist": {}, "gap": "x"}]
    assert route_after_reflection(base_state) == "formatter"


def test_route_after_reflection_exits_on_error_regardless_of_cycle_count(base_state):
    base_state["cycle_count"] = 0
    base_state["error"] = "synthesis_failed: timeout"
    base_state["reflection_history"] = []
    assert route_after_reflection(base_state) == "formatter"


def test_route_after_reflection_defaults_to_pass_with_empty_history(base_state):
    base_state["cycle_count"] = 1
    base_state["reflection_history"] = []
    assert route_after_reflection(base_state) == "formatter"
