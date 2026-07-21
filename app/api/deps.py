"""
Auth dependency — REAL JWT verification against Supabase.

Matches the design from the auth-flow scrutiny session: every
authenticated route depends on get_current_user_id(), which extracts
and verifies the bearer token, returning the JWT's `sub` claim as
user_id. This is the ONLY source of user_id anywhere in the request
lifecycle — it is never read from the request body, matching the
"namespace must come from the verified JWT, never client input"
security rule from our design.

Supabase issues JWTs signed with one of two schemes depending on
project age/settings (see app/api/jwks.py for details):
  - HS256 (legacy): verified against a shared secret, SUPABASE_JWT_SECRET.
  - ES256 / RS256 (current default since Oct 2025): verified against
    the project's public JWKS, fetched and cached from
    {SUPABASE_URL}/auth/v1/.well-known/jwks.json.

This function checks the token's own `alg` header and verifies
accordingly — a project could be using either, and there's no way to
know in advance which one without inspecting the token.

There is deliberately NO fallback that accepts an unverified token.
If SUPABASE_JWT_SECRET/SUPABASE_URL aren't configured, this fails
loudly (500, AuthConfigError) rather than silently accepting anything
— the previous version of this file did the latter, which meant any
non-empty string was accepted as a valid token. That was the
project's single most serious security hole; see the README for how
it was found and fixed.
"""

from __future__ import annotations

import os

import jwt
from fastapi import Header, HTTPException

from app.api.jwks import AuthConfigError, find_key, get_jwks

JWT_AUDIENCE = "authenticated"

# Algorithms Supabase Auth issues tokens with. Anything else in a
# token's `alg` header is rejected outright — this is a strict
# allowlist, not an attempt to support every algorithm PyJWT knows.
SYMMETRIC_ALGORITHMS = {"HS256"}
ASYMMETRIC_ALGORITHMS = {"ES256", "RS256"}


async def verify_jwt(token: str) -> str:
    """Verifies a Supabase-issued JWT and returns its `sub` claim
    (the authenticated user's UUID). Raises ValueError on any
    verification failure (bad signature, expired, wrong audience,
    malformed, missing claim) — the caller converts that to a 401.
    Raises AuthConfigError if the server itself isn't configured to
    verify tokens at all — the caller converts that to a 500, since
    it's an operator error, not something the client did wrong.
    """
    if not token:
        raise ValueError("empty token")

    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as e:
        raise ValueError(f"malformed token header: {e}") from e

    alg = header.get("alg")

    try:
        if alg in SYMMETRIC_ALGORITHMS:
            payload = await _verify_symmetric(token, alg)
        elif alg in ASYMMETRIC_ALGORITHMS:
            payload = await _verify_asymmetric(token, alg, header.get("kid"))
        else:
            raise ValueError(f"unsupported or missing JWT algorithm: {alg!r}")
    except AuthConfigError:
        raise
    except jwt.PyJWTError as e:
        raise ValueError(f"token verification failed: {e}") from e

    sub = payload.get("sub")
    if not sub:
        raise ValueError("token missing 'sub' claim")
    return sub


async def _verify_symmetric(token: str, alg: str) -> dict:
    secret = os.environ.get("SUPABASE_JWT_SECRET")
    if not secret:
        raise AuthConfigError("SUPABASE_JWT_SECRET is not configured")
    return jwt.decode(token, secret, algorithms=[alg], audience=JWT_AUDIENCE)


async def _verify_asymmetric(token: str, alg: str, kid: str | None) -> dict:
    jwks = await get_jwks()
    matching_key = find_key(jwks, kid)

    if matching_key is None:
        # Expected right after Supabase rotates its signing key —
        # refetch once, bypassing the cache, before giving up.
        jwks = await get_jwks(force_refresh=True)
        matching_key = find_key(jwks, kid)

    if matching_key is None:
        raise ValueError(f"no matching JWKS key for kid={kid!r}")

    signing_key = jwt.PyJWK(matching_key).key
    return jwt.decode(token, signing_key, algorithms=[alg], audience=JWT_AUDIENCE)


async def get_current_user_id(authorization: str = Header(default="")) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")

    token = authorization.removeprefix("Bearer ").strip()

    try:
        return await verify_jwt(token)
    except AuthConfigError as e:
        raise HTTPException(status_code=500, detail=f"Auth misconfigured: {e}") from e
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}") from e
