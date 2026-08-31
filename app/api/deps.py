from __future__ import annotations

import os

import jwt
from fastapi import Header, HTTPException

from app.api.jwks import AuthConfigError, find_key, get_jwks

JWT_AUDIENCE = "authenticated"

SYMMETRIC_ALGORITHMS = {"HS256"}
ASYMMETRIC_ALGORITHMS = {"ES256", "RS256"}


async def verify_jwt(token: str) -> str:
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
