"""
Async JWKS (JSON Web Key Set) fetching and caching for Supabase's
asymmetric JWT verification path (ES256/RS256).

Supabase moved to asymmetric JWT signing by default for all new
projects starting October 2025 — the public verification key is
served from `{SUPABASE_URL}/auth/v1/.well-known/jwks.json`, and
verification happens entirely on our side (no round trip to Supabase
per request). Older projects may still use the legacy symmetric
HS256 scheme with a shared secret instead — see deps.py for how both
paths are selected based on the token's own `alg` header.

Cached for 600 seconds, matching Supabase's own documented edge-cache
TTL for this endpoint (their docs explicitly warn against caching
longer than that, since it would delay picking up a rotated key).
"""

from __future__ import annotations

import os
import time

import httpx

JWKS_CACHE_TTL_SECONDS = 600

_jwks_cache: dict | None = None
_jwks_cache_fetched_at: float = 0.0


class AuthConfigError(RuntimeError):
    """Raised when the server itself is misconfigured (no
    SUPABASE_URL set) — this is an operator error, not a client
    error, and should surface as a 500, not a 401. Deliberately a
    distinct exception type from the ValueError used for genuine
    token-verification failures, so callers can tell the two apart."""


async def get_jwks(*, force_refresh: bool = False) -> dict:
    """Returns the cached JWKS document, refetching if the cache is
    empty, stale, or force_refresh is requested (used once, as a
    fallback, when a token's `kid` isn't found in the cached set —
    this is the normal, expected behavior right after Supabase
    rotates its signing key, not an error condition by itself)."""
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
