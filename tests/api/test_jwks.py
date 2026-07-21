"""
Tests for app.api.jwks — the JWKS fetching and caching logic itself,
isolated from deps.py. test_deps.py's asymmetric-path tests
monkeypatch deps.get_jwks() directly, which proves deps.py calls it
correctly but never exercises get_jwks()'s own fetch/cache/refresh
logic — these tests close that gap using httpx.MockTransport, the
same never-touch-the-real-network pattern used in test_loaders.py.
"""

from __future__ import annotations

import time

import httpx
import pytest

import app.api.jwks as jwks_module
from app.api.jwks import AuthConfigError, find_key, get_jwks


@pytest.fixture(autouse=True)
def reset_jwks_cache(monkeypatch):
    """jwks.py's cache is module-level global state — reset it
    before every test so one test's fetch can't leak into another's
    assertions about whether a fetch happened."""
    monkeypatch.setattr(jwks_module, "_jwks_cache", None)
    monkeypatch.setattr(jwks_module, "_jwks_cache_fetched_at", 0.0)


def _patch_jwks_response(monkeypatch, *, keys: list[dict], call_counter: list[int] | None = None):
    async def handler(request):
        if call_counter is not None:
            call_counter.append(1)
        return httpx.Response(200, json={"keys": keys})

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs.pop("transport", None)
            super().__init__(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(jwks_module.httpx, "AsyncClient", PatchedAsyncClient)


async def test_get_jwks_fetches_and_returns_keys(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    _patch_jwks_response(monkeypatch, keys=[{"kid": "key-1", "kty": "EC"}])

    jwks = await get_jwks()

    assert jwks == {"keys": [{"kid": "key-1", "kty": "EC"}]}


async def test_get_jwks_requests_the_correct_well_known_url(monkeypatch):
    requested_urls = []

    async def handler(request):
        requested_urls.append(str(request.url))
        return httpx.Response(200, json={"keys": []})

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs.pop("transport", None)
            super().__init__(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(jwks_module.httpx, "AsyncClient", PatchedAsyncClient)
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")

    await get_jwks()

    assert requested_urls == ["https://project.supabase.co/auth/v1/.well-known/jwks.json"]


async def test_get_jwks_raises_auth_config_error_when_url_not_set(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)

    with pytest.raises(AuthConfigError, match="SUPABASE_URL is not configured"):
        await get_jwks()


async def test_get_jwks_uses_cache_on_second_call_within_ttl(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    call_counter: list[int] = []
    _patch_jwks_response(monkeypatch, keys=[{"kid": "key-1"}], call_counter=call_counter)

    await get_jwks()
    await get_jwks()  # should hit cache, not fetch again

    assert len(call_counter) == 1


async def test_get_jwks_refetches_after_cache_expires(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    call_counter: list[int] = []
    _patch_jwks_response(monkeypatch, keys=[{"kid": "key-1"}], call_counter=call_counter)

    await get_jwks()

    # Simulate time passing beyond the cache TTL by rewinding the
    # recorded fetch timestamp, rather than actually sleeping in a test.
    monkeypatch.setattr(
        jwks_module, "_jwks_cache_fetched_at", time.monotonic() - jwks_module.JWKS_CACHE_TTL_SECONDS - 1
    )

    await get_jwks()

    assert len(call_counter) == 2


async def test_get_jwks_force_refresh_bypasses_a_fresh_cache(monkeypatch):
    """The exact scenario this exists for: a token's kid isn't found
    in the cached set right after Supabase rotates its signing key —
    force_refresh must bypass even a cache that's still within its
    normal TTL."""
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    call_counter: list[int] = []
    _patch_jwks_response(monkeypatch, keys=[{"kid": "key-1"}], call_counter=call_counter)

    await get_jwks()
    await get_jwks(force_refresh=True)

    assert len(call_counter) == 2


async def test_get_jwks_raises_on_http_error(monkeypatch):
    async def handler(request):
        return httpx.Response(500, text="Internal Server Error")

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs.pop("transport", None)
            super().__init__(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(jwks_module.httpx, "AsyncClient", PatchedAsyncClient)
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")

    with pytest.raises(httpx.HTTPStatusError):
        await get_jwks()


# ---- find_key ----

def test_find_key_returns_matching_key():
    jwks = {"keys": [{"kid": "a"}, {"kid": "b"}]}
    assert find_key(jwks, "b") == {"kid": "b"}


def test_find_key_returns_none_when_no_match():
    jwks = {"keys": [{"kid": "a"}]}
    assert find_key(jwks, "nonexistent") is None


def test_find_key_returns_none_for_empty_keys():
    assert find_key({"keys": []}, "any-kid") is None


def test_find_key_handles_missing_keys_field_defensively():
    assert find_key({}, "any-kid") is None
