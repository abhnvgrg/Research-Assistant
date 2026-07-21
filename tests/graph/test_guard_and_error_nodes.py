"""
quota_check_node, formatter_node, and error_handler_node tests.

quota_check_node is where the "auth fix" pattern repeats for cost
control: the old version always returned quota_ok=True regardless of
anything, exactly as fake as the old verify_jwt() stub. These tests
prove real enforcement — a user who has exhausted their quota is
genuinely blocked from starting a new run, and usage is genuinely
recorded (by formatter_node and error_handler_node) after a run
completes, which is what makes the check meaningful in the first
place rather than checking a number that never changes.
"""

from __future__ import annotations

from app.graph.nodes import error_handler_node, formatter_node, quota_check_node


async def test_quota_check_node_allows_user_with_no_prior_usage(base_state, fresh_quota_store):
    result = await quota_check_node(base_state)

    assert result["quota_ok"] is True
    assert result["error"] is None


async def test_quota_check_node_blocks_user_who_has_exceeded_quota(base_state, fresh_quota_store):
    """The core enforcement guarantee — a user already at their
    limit must be blocked BEFORE any LLM call is made for the new
    run, not partway through one."""
    await fresh_quota_store.set_limit(base_state["user_id"], 1000)
    await fresh_quota_store.add_usage(base_state["user_id"], 1000)

    result = await quota_check_node(base_state)

    assert result["quota_ok"] is False
    assert "quota_exceeded" in result["error"]
    assert "1000" in result["error"]  # tokens_used and quota_limit both surfaced in the message


async def test_quota_check_node_allows_user_just_under_limit(base_state, fresh_quota_store):
    await fresh_quota_store.set_limit(base_state["user_id"], 1000)
    await fresh_quota_store.add_usage(base_state["user_id"], 999)

    result = await quota_check_node(base_state)

    assert result["quota_ok"] is True


async def test_quota_check_node_isolates_usage_per_user(base_state, fresh_quota_store):
    """Exhausting one user's quota must never affect another user's
    ability to start a run — the same isolation guarantee QuotaStore
    itself already proves, checked again here at the node level."""
    await fresh_quota_store.set_limit("some-other-user", 100)
    await fresh_quota_store.add_usage("some-other-user", 100)

    result = await quota_check_node(base_state)  # base_state's user_id is different

    assert result["quota_ok"] is True


async def test_formatter_node_records_real_usage_against_quota(base_state, fresh_quota_store):
    """This is what makes quota_check_node meaningful in the first
    place — without this, 'checking quota' would just be checking a
    number that never changes, exactly as fake as the old stub."""
    base_state["tokens_used"] = 4200

    await formatter_node(base_state)

    usage = await fresh_quota_store.get_usage(base_state["user_id"])
    assert usage.tokens_used == 4200


async def test_formatter_node_recording_failure_does_not_break_the_response(
    base_state, fresh_quota_store, monkeypatch
):
    """Recording usage must never itself crash the run — worst case
    on failure is under-billed usage, not a broken user-facing
    result. Simulates QuotaStore.add_usage raising, patched on the
    SAME fresh store instance get_quota_store() is already returning
    for this test (see the fresh_quota_store fixture)."""

    async def broken_add_usage(user_id, tokens):
        raise RuntimeError("simulated store failure")

    monkeypatch.setattr(fresh_quota_store, "add_usage", broken_add_usage)

    result = await formatter_node(base_state)

    assert "mlflow_run_id" in result  # node still completed normally


async def test_error_handler_node_still_records_usage_before_a_run_failed(base_state, fresh_quota_store):
    """A rejected/failed run may still have made real, billable LLM
    calls before failing (e.g. one decomposer call before a
    topic-less query was rejected) — that spend must still count
    against quota, not be silently forgotten because the run never
    reached formatter_node."""
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
    """Defensive: error_handler_node may be reached via a path that
    didn't actually set state['error'] (shouldn't happen given our
    conditional edges, but the node must not crash if it does)."""
    result = await error_handler_node(base_state)
    assert "unknown error" in result["answer"]
