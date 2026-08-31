from __future__ import annotations

from app.quota_store import DEFAULT_QUOTA_LIMIT, QuotaStore


async def test_new_user_starts_with_zero_usage_and_default_limit():
    store = QuotaStore()
    usage = await store.get_usage("user-a")

    assert usage.tokens_used == 0
    assert usage.quota_limit == DEFAULT_QUOTA_LIMIT


async def test_new_user_has_quota_by_default():
    store = QuotaStore()
    assert await store.has_quota("user-a") is True


async def test_add_usage_increments_tokens_used():
    store = QuotaStore()
    await store.add_usage("user-a", 500)
    await store.add_usage("user-a", 300)

    usage = await store.get_usage("user-a")
    assert usage.tokens_used == 800


async def test_has_quota_returns_false_once_limit_reached():
    store = QuotaStore()
    await store.set_limit("user-a", 1000)
    await store.add_usage("user-a", 1000)

    assert await store.has_quota("user-a") is False


async def test_has_quota_returns_false_once_limit_exceeded():
    store = QuotaStore()
    await store.set_limit("user-a", 1000)
    await store.add_usage("user-a", 1500)

    assert await store.has_quota("user-a") is False


async def test_has_quota_returns_true_just_under_limit():
    store = QuotaStore()
    await store.set_limit("user-a", 1000)
    await store.add_usage("user-a", 999)

    assert await store.has_quota("user-a") is True


async def test_usage_is_isolated_per_user():
    store = QuotaStore()
    await store.add_usage("user-a", 5000)

    usage_a = await store.get_usage("user-a")
    usage_b = await store.get_usage("user-b")

    assert usage_a.tokens_used == 5000
    assert usage_b.tokens_used == 0
    assert await store.has_quota("user-b") is True


async def test_add_usage_ignores_zero_or_negative_deltas():
    store = QuotaStore()
    await store.add_usage("user-a", 0)
    await store.add_usage("user-a", -100)

    usage = await store.get_usage("user-a")
    assert usage.tokens_used == 0


async def test_set_limit_changes_quota_for_future_checks():
    store = QuotaStore()
    await store.add_usage("user-a", 500)
    assert await store.has_quota("user-a") is True

    await store.set_limit("user-a", 400)
    assert await store.has_quota("user-a") is False
