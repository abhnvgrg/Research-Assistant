"""
In-memory quota store — tracks real per-user token consumption
against a quota limit, mirroring the `users` table columns from the
Supabase schema design (tokens_used, quota_limit).

Same swappable-seam principle as RunStore/JobStore: a real deployment
replaces this with a Supabase-backed implementation behind the same
interface. Unlike the old quota_check_node (which always returned
quota_ok=True regardless of anything), this store is only meaningful
because real token usage is actually recorded here after every run —
see app/graph/nodes.py's formatter_node and the tokens_used reducer
in app/graph/state.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_QUOTA_LIMIT = 200_000  # matches users.quota_limit default in supabase_schema.sql


@dataclass
class UserQuota:
    user_id: str
    tokens_used: int = 0
    quota_limit: int = DEFAULT_QUOTA_LIMIT


class QuotaStore:
    def __init__(self) -> None:
        self._quotas: dict[str, UserQuota] = {}

    def _get_or_create(self, user_id: str) -> UserQuota:
        if user_id not in self._quotas:
            self._quotas[user_id] = UserQuota(user_id=user_id)
        return self._quotas[user_id]

    async def has_quota(self, user_id: str) -> bool:
        """async for interface parity with SupabaseQuotaStore, which
        genuinely needs a network round trip — see
        RunStore.get_result's docstring for the same reasoning.
        Checked by quota_check_node BEFORE any LLM call is made for a
        run — a user already at or over their limit is blocked from
        starting a new run entirely, not just warned partway through
        one."""
        quota = self._get_or_create(user_id)
        return quota.tokens_used < quota.quota_limit

    async def add_usage(self, user_id: str, tokens: int) -> None:
        """Called once per completed run (successful or failed) with
        the REAL total tokens consumed across every LLM call in that
        run — see state["tokens_used"], which accumulates via an
        operator.add reducer across every node that makes an LLM
        call. Negative/zero deltas are ignored defensively."""
        if tokens <= 0:
            return
        quota = self._get_or_create(user_id)
        quota.tokens_used += tokens

    async def get_usage(self, user_id: str) -> UserQuota:
        return self._get_or_create(user_id)

    async def set_limit(self, user_id: str, limit: int) -> None:
        """Test/admin hook — a real deployment would expose this via
        a plan-upgrade flow writing to Supabase, not a public API."""
        quota = self._get_or_create(user_id)
        quota.quota_limit = limit


# Module-level singleton — a real deployment would replace this with
# a Supabase-backed implementation behind the same interface.
quota_store = QuotaStore()


class SupabaseQuotaStore:
    """Postgres-backed implementation, querying the `users` table
    directly (tokens_used, quota_limit columns — see
    supabase_schema.sql). Every method signature matches QuotaStore
    exactly, so callers work unmodified against either.

    Unlike RunStore, quota has no live-streaming concern at all —
    every operation is a simple read or write, so this can be fully
    Postgres-backed with no in-memory hybrid needed.

    A user row is expected to already exist (created at signup, per
    the auth design) — has_quota()/get_usage() on a genuinely unknown
    user_id return the same defaults a fresh QuotaStore would rather
    than raising, since that's how they'd behave for a new signup
    whose users row insert raced with their first request.
    """

    def __init__(self, pool) -> None:
        self._pool = pool

    async def has_quota(self, user_id: str) -> bool:
        row = await self._pool.fetchrow(
            "SELECT tokens_used, quota_limit FROM users WHERE id = $1", user_id
        )
        if row is None:
            return True  # unknown user — treat as fresh, same as QuotaStore
        return row["tokens_used"] < row["quota_limit"]

    async def add_usage(self, user_id: str, tokens: int) -> None:
        if tokens <= 0:
            return
        await self._pool.execute(
            """
            INSERT INTO users (id, tokens_used)
            VALUES ($1, $2)
            ON CONFLICT (id) DO UPDATE
              SET tokens_used = users.tokens_used + EXCLUDED.tokens_used
            """,
            user_id,
            tokens,
        )

    async def get_usage(self, user_id: str) -> UserQuota:
        row = await self._pool.fetchrow(
            "SELECT tokens_used, quota_limit FROM users WHERE id = $1", user_id
        )
        if row is None:
            return UserQuota(user_id=user_id)
        return UserQuota(user_id=user_id, tokens_used=row["tokens_used"], quota_limit=row["quota_limit"])

    async def set_limit(self, user_id: str, limit: int) -> None:
        await self._pool.execute(
            """
            INSERT INTO users (id, quota_limit)
            VALUES ($1, $2)
            ON CONFLICT (id) DO UPDATE SET quota_limit = EXCLUDED.quota_limit
            """,
            user_id,
            limit,
        )


_supabase_quota_store: SupabaseQuotaStore | None = None


def get_quota_store():
    """Returns the Supabase-backed store if app.db's connection pool
    initialized successfully, otherwise falls back to the in-memory
    QuotaStore — the same fallback pattern as
    app.graph.checkpointing.get_research_graph(). Callers (nodes.py)
    should call this instead of importing `quota_store` directly, so
    they transparently get real persistence once the DB is available
    without needing to know which backend is active."""
    global _supabase_quota_store
    from app.db import get_pool, is_db_enabled

    if not is_db_enabled():
        return quota_store

    if _supabase_quota_store is None:
        _supabase_quota_store = SupabaseQuotaStore(get_pool())
    return _supabase_quota_store
