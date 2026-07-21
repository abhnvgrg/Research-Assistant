from __future__ import annotations

import logging
import os

import asyncpg

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def init_pool() -> None:
    """Best-effort DB initialization.

    If DATABASE_URL is missing or the connection fails, we keep the app
    running and let callers transparently fall back to in-memory stores.
    """
    global _pool

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.info("DATABASE_URL not configured; using in-memory stores")
        _pool = None
        return

    try:
        _pool = await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=5)
    except Exception as exc:  # pragma: no cover - defensive startup fallback
        logger.warning("Database pool initialization failed; using in-memory stores: %s", exc)
        _pool = None


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def is_db_enabled() -> bool:
    return _pool is not None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool is not initialized")
    return _pool

