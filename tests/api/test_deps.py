"""
Unit tests for app.api.deps — REAL JWT verification, isolated from
FastAPI's request/response cycle.

This file exists specifically to prove the auth security fix works:
the previous version of app/api/deps.py accepted ANY non-empty
string as a valid bearer token (a stub, clearly documented as such,
but still a genuine hole if ever deployed as-is). These tests prove
that hole is closed — a tampered signature, wrong secret, expired
token, wrong audience, or malformed token is all correctly rejected,
and only a token genuinely signed with the configured secret (or a
genuinely matching JWKS key, for the asymmetric path) is accepted.
"""

from __future__ import annotations

import json
import time

import jwt
import pytest
from fastapi import HTTPException

from app.api.deps import get_current_user_id, verify_jwt
from app.api.jwks import AuthConfigError
from tests.api.conftest import TEST_JWT_SECRET


def _valid_payload(sub: str = "user-abc-123") -> dict:
    return {"sub": sub, "aud": "authenticated", "exp": int(time.time()) + 3600}


def _sign(payload: dict, secret: str = TEST_JWT_SECRET, alg: str = "HS256") -> str:
    return jwt.encode(payload, secret, algorithm=alg)


# ---- the core security guarantee ----

async def test_verify_jwt_accepts_a_genuinely_valid_token():
    token = _sign(_valid_payload("user-abc-123"))
    assert await verify_jwt(token) == "user-abc-123"


async def test_verify_jwt_rejects_a_token_signed_with_the_wrong_secret():
    """The core security fix, proven directly: a token that LOOKS
    correctly formed (right claims, right structure) but was signed
    with a different secret than the server is configured with must
    be rejected. This is exactly the attack the old stub was
    vulnerable to — any string was accepted, signed or not."""
    token = _sign(_valid_payload(), secret="attacker-controlled-secret")

    with pytest.raises(ValueError, match="verification failed"):
        await verify_jwt(token)


async def test_verify_jwt_rejects_an_arbitrary_non_jwt_string():
    """The exact case the old stub incorrectly accepted: a plain,
    unsigned, non-JWT string is not a valid token at all."""
    with pytest.raises(ValueError, match="malformed token header"):
        await verify_jwt("just-some-random-string-not-a-jwt")


async def test_verify_jwt_rejects_empty_token():
    with pytest.raises(ValueError, match="empty token"):
        await verify_jwt("")


async def test_verify_jwt_rejects_expired_token():
    expired_payload = {
        "sub": "user-abc-123",
        "aud": "authenticated",
        "exp": int(time.time()) - 60,  # 60 seconds in the past
    }
    token = _sign(expired_payload)

    with pytest.raises(ValueError, match="verification failed"):
        await verify_jwt(token)


async def test_verify_jwt_rejects_wrong_audience():
    """Supabase tokens carry aud='authenticated' — a token issued
    for a different audience (e.g. a service-role token, or a token
    from an entirely different system) must not be accepted here."""
    payload = {**_valid_payload(), "aud": "some-other-audience"}
    token = _sign(payload)

    with pytest.raises(ValueError, match="verification failed"):
        await verify_jwt(token)


async def test_verify_jwt_rejects_token_missing_sub_claim():
    payload = {"aud": "authenticated", "exp": int(time.time()) + 3600}  # no 'sub'
    token = _sign(payload)

    with pytest.raises(ValueError, match="missing 'sub' claim"):
        await verify_jwt(token)


async def test_verify_jwt_rejects_unsupported_algorithm():
    """'none' algorithm tokens (unsigned) are a classic JWT
    vulnerability — must be rejected outright, not silently accepted
    because there's technically no signature to fail verification."""
    # PyJWT itself refuses to encode with alg="none" without an
    # explicit opt-in, which is itself a good sign — construct the
    # token by hand to simulate an attacker who isn't using PyJWT.
    import base64

    header_b64 = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps(_valid_payload()).encode()).rstrip(b"=")
    unsigned_token = f"{header_b64.decode()}.{payload_b64.decode()}."

    with pytest.raises(ValueError, match="unsupported or missing JWT algorithm"):
        await verify_jwt(unsigned_token)


async def test_verify_jwt_raises_auth_config_error_when_secret_not_configured(monkeypatch):
    """Distinct from a bad TOKEN: if the server itself isn't
    configured with a secret, that's an operator error and must be
    surfaced distinctly (as AuthConfigError -> 500), not silently
    treated as 'any token is fine' — which was the old behavior."""
    monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
    token = _sign(_valid_payload())

    with pytest.raises(AuthConfigError):
        await verify_jwt(token)


# ---- get_current_user_id — the FastAPI dependency wrapper ----

async def test_get_current_user_id_accepts_valid_bearer_header():
    token = _sign(_valid_payload("user-xyz"))
    user_id = await get_current_user_id(authorization=f"Bearer {token}")
    assert user_id == "user-xyz"


async def test_get_current_user_id_rejects_missing_header():
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user_id(authorization="")
    assert exc_info.value.status_code == 401


async def test_get_current_user_id_rejects_non_bearer_scheme():
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user_id(authorization="Basic dXNlcjpwYXNz")
    assert exc_info.value.status_code == 401


async def test_get_current_user_id_rejects_bearer_with_empty_token():
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user_id(authorization="Bearer ")
    assert exc_info.value.status_code == 401


async def test_get_current_user_id_rejects_tampered_token_with_401():
    token = _sign(_valid_payload(), secret="wrong-secret")
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user_id(authorization=f"Bearer {token}")
    assert exc_info.value.status_code == 401


async def test_get_current_user_id_surfaces_config_error_as_500(monkeypatch):
    monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
    token = _sign(_valid_payload())

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user_id(authorization=f"Bearer {token}")
    assert exc_info.value.status_code == 500


# ---- asymmetric (ES256/JWKS) path ----

async def test_verify_jwt_accepts_valid_es256_token_via_jwks(monkeypatch):
    """Proves the asymmetric path end-to-end: generates a real EC
    keypair, signs a token with the private key, serves the public
    key as a JWKS document (mocked — no real network call), and
    confirms verification succeeds using only the public key. This
    is the current default for all new Supabase projects since
    October 2025."""
    from cryptography.hazmat.primitives.asymmetric import ec
    from jwt.algorithms import ECAlgorithm

    private_key = ec.generate_private_key(ec.SECP256R1())
    kid = "test-key-1"

    token = jwt.encode(
        _valid_payload("es256-user"),
        private_key,
        algorithm="ES256",
        headers={"kid": kid},
    )

    algo = ECAlgorithm(ECAlgorithm.SHA256)
    public_jwk_dict = json.loads(algo.to_jwk(private_key.public_key()))
    public_jwk_dict["kid"] = kid

    import app.api.deps as deps_module

    async def fake_get_jwks(*, force_refresh: bool = False) -> dict:
        return {"keys": [public_jwk_dict]}

    monkeypatch.setattr(deps_module, "get_jwks", fake_get_jwks)

    result = await verify_jwt(token)
    assert result == "es256-user"


async def test_verify_jwt_rejects_es256_token_with_unknown_kid(monkeypatch):
    """A kid that doesn't match anything in the JWKS — e.g. an
    attacker-forged token, or a genuinely rotated key we haven't
    refetched — must be rejected, not silently accepted."""
    from cryptography.hazmat.primitives.asymmetric import ec

    private_key = ec.generate_private_key(ec.SECP256R1())
    token = jwt.encode(
        _valid_payload(), private_key, algorithm="ES256", headers={"kid": "unknown-kid"}
    )

    import app.api.deps as deps_module

    async def fake_get_jwks(*, force_refresh: bool = False) -> dict:
        return {"keys": []}  # empty — no key will ever match

    monkeypatch.setattr(deps_module, "get_jwks", fake_get_jwks)

    with pytest.raises(ValueError, match="no matching JWKS key"):
        await verify_jwt(token)
