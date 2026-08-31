from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_QUOTA_LIMIT = 200_000


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
        quota = self._get_or_create(user_id)
        return quota.tokens_used < quota.quota_limit

    async def add_usage(self, user_id: str, tokens: int) -> None:
        if tokens <= 0:
            return
        quota = self._get_or_create(user_id)
        quota.tokens_used += tokens

    async def get_usage(self, user_id: str) -> UserQuota:
        return self._get_or_create(user_id)

    async def set_limit(self, user_id: str, limit: int) -> None:
        quota = self._get_or_create(user_id)
        quota.quota_limit = limit


quota_store = QuotaStore()


class SupabaseQuotaStore:
    def __init__(self, pool) -> None:
        self._pool = pool

    async def has_quota(self, user_id: str) -> bool:
        row = await self._pool.fetchrow(
            "SELECT tokens_used, quota_limit FROM users WHERE id = $1", user_id
        )
        if row is None:
            return True
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
    global _supabase_quota_store
    from app.db import get_pool, is_db_enabled

    if not is_db_enabled():
        return quota_store

    if _supabase_quota_store is None:
        _supabase_quota_store = SupabaseQuotaStore(get_pool())
    return _supabase_quota_store
