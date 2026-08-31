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


async def test_verify_jwt_accepts_a_genuinely_valid_token():
    token = _sign(_valid_payload("user-abc-123"))
    assert await verify_jwt(token) == "user-abc-123"


async def test_verify_jwt_rejects_a_token_signed_with_the_wrong_secret():
    token = _sign(_valid_payload(), secret="attacker-controlled-secret")

    with pytest.raises(ValueError, match="verification failed"):
        await verify_jwt(token)


async def test_verify_jwt_rejects_an_arbitrary_non_jwt_string():
    with pytest.raises(ValueError, match="malformed token header"):
        await verify_jwt("just-some-random-string-not-a-jwt")


async def test_verify_jwt_rejects_empty_token():
    with pytest.raises(ValueError, match="empty token"):
        await verify_jwt("")


async def test_verify_jwt_rejects_expired_token():
    expired_payload = {
        "sub": "user-abc-123",
        "aud": "authenticated",
        "exp": int(time.time()) - 60,
    }
    token = _sign(expired_payload)

    with pytest.raises(ValueError, match="verification failed"):
        await verify_jwt(token)


async def test_verify_jwt_rejects_wrong_audience():
    payload = {**_valid_payload(), "aud": "some-other-audience"}
    token = _sign(payload)

    with pytest.raises(ValueError, match="verification failed"):
        await verify_jwt(token)


async def test_verify_jwt_rejects_token_missing_sub_claim():
    payload = {"aud": "authenticated", "exp": int(time.time()) + 3600}
    token = _sign(payload)

    with pytest.raises(ValueError, match="missing 'sub' claim"):
        await verify_jwt(token)


async def test_verify_jwt_rejects_unsupported_algorithm():
    import base64

    header_b64 = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps(_valid_payload()).encode()).rstrip(b"=")
    unsigned_token = f"{header_b64.decode()}.{payload_b64.decode()}."

    with pytest.raises(ValueError, match="unsupported or missing JWT algorithm"):
        await verify_jwt(unsigned_token)


async def test_verify_jwt_raises_auth_config_error_when_secret_not_configured(monkeypatch):
    monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
    token = _sign(_valid_payload())

    with pytest.raises(AuthConfigError):
        await verify_jwt(token)


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


async def test_verify_jwt_accepts_valid_es256_token_via_jwks(monkeypatch):
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
    from cryptography.hazmat.primitives.asymmetric import ec

    private_key = ec.generate_private_key(ec.SECP256R1())
    token = jwt.encode(
        _valid_payload(), private_key, algorithm="ES256", headers={"kid": "unknown-kid"}
    )

    import app.api.deps as deps_module

    async def fake_get_jwks(*, force_refresh: bool = False) -> dict:
        return {"keys": []}

    monkeypatch.setattr(deps_module, "get_jwks", fake_get_jwks)

    with pytest.raises(ValueError, match="no matching JWKS key"):
        await verify_jwt(token)
