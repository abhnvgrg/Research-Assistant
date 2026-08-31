from __future__ import annotations

import os
import time

import httpx

JWKS_CACHE_TTL_SECONDS = 600

_jwks_cache: dict | None = None
_jwks_cache_fetched_at: float = 0.0


class AuthConfigError(RuntimeError):
    pass


async def get_jwks(*, force_refresh: bool = False) -> dict:
    global _jwks_cache, _jwks_cache_fetched_at

    now = time.monotonic()
    cache_is_fresh = _jwks_cache is not None and (now - _jwks_cache_fetched_at) < JWKS_CACHE_TTL_SECONDS
    if cache_is_fresh and not force_refresh:
        return _jwks_cache

    supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    if not supabase_url:
        raise AuthConfigError("SUPABASE_URL is not configured — cannot fetch JWKS")

    jwks_url = f"{supabase_url}/auth/v1/.well-known/jwks.json"

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(jwks_url)
        response.raise_for_status()
        _jwks_cache = response.json()
        _jwks_cache_fetched_at = now

    return _jwks_cache


def find_key(jwks: dict, kid: str | None) -> dict | None:
    return next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
