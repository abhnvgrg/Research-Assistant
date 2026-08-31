from __future__ import annotations

from app.graph.nodes import error_handler_node, formatter_node, quota_check_node


async def test_quota_check_node_allows_user_with_no_prior_usage(base_state, fresh_quota_store):
    result = await quota_check_node(base_state)

    assert result["quota_ok"] is True
    assert result["error"] is None


async def test_quota_check_node_blocks_user_who_has_exceeded_quota(base_state, fresh_quota_store):
    await fresh_quota_store.set_limit(base_state["user_id"], 1000)
    await fresh_quota_store.add_usage(base_state["user_id"], 1000)

    result = await quota_check_node(base_state)

    assert result["quota_ok"] is False
    assert "quota_exceeded" in result["error"]
    assert "1000" in result["error"]


async def test_quota_check_node_allows_user_just_under_limit(base_state, fresh_quota_store):
    await fresh_quota_store.set_limit(base_state["user_id"], 1000)
    await fresh_quota_store.add_usage(base_state["user_id"], 999)

    result = await quota_check_node(base_state)

    assert result["quota_ok"] is True


async def test_quota_check_node_isolates_usage_per_user(base_state, fresh_quota_store):
    await fresh_quota_store.set_limit("some-other-user", 100)
    await fresh_quota_store.add_usage("some-other-user", 100)

    result = await quota_check_node(base_state)

    assert result["quota_ok"] is True


async def test_formatter_node_records_real_usage_against_quota(base_state, fresh_quota_store):
    base_state["tokens_used"] = 4200

    await formatter_node(base_state)

    usage = await fresh_quota_store.get_usage(base_state["user_id"])
    assert usage.tokens_used == 4200


async def test_formatter_node_recording_failure_does_not_break_the_response(
    base_state, fresh_quota_store, monkeypatch
):
    async def broken_add_usage(user_id, tokens):
        raise RuntimeError("simulated store failure")

    monkeypatch.setattr(fresh_quota_store, "add_usage", broken_add_usage)

    result = await formatter_node(base_state)

    assert "mlflow_run_id" in result


async def test_error_handler_node_still_records_usage_before_a_run_failed(base_state, fresh_quota_store):
    base_state["tokens_used"] = 75
    base_state["error"] = "decomposition_failed: some LLM error"

    await error_handler_node(base_state)

    usage = await fresh_quota_store.get_usage(base_state["user_id"])
    assert usage.tokens_used == 75


async def test_error_handler_node_formats_known_error(base_state, fresh_quota_store):
    base_state["error"] = "synthesis_failed: OpenAI 503"
    result = await error_handler_node(base_state)

    assert "synthesis_failed: OpenAI 503" in result["answer"]
    assert result["citations"] == {}


async def test_error_handler_node_handles_missing_error_gracefully(base_state, fresh_quota_store):
    result = await error_handler_node(base_state)
    assert "unknown error" in result["answer"]
